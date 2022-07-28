from __future__ import annotations as __annotations__ # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi

import typing
if typing.TYPE_CHECKING:
    from typing import Tuple

class _AttachedReparameterizeOp(dr.CustomOp):
    """
    Dr.Jit custom operation that reparameterizes rays generated via
    attached sampling strategies. Based on the paper

      "Monte Carlo Estimators for Differential Light Transport"
      (Proceedings of SIGGRAPH'21) by Tizan Zeltner, Sébastien Speierer,
      Iliyan Georgiev, and Wenzel Jakob.

    This is needed to to avoid bias caused by the discontinuous visibility
    function in gradient-based optimization involving attached samples.
    """
    def eval(self, scene, rng, wo, pdf, si,
             num_rays, kappa, exponent, antithetic, naive, active):
        # Stash all of this information for the forward/backward passes
        self.scene = scene
        self.rng = rng
        self.wo = wo
        self.pdf = pdf
        self.si = si
        self.num_rays = num_rays
        self.kappa = kappa
        self.exponent = exponent
        self.antithetic = antithetic
        self.naive = naive
        self.active = active

        # The reparameterization is simply the identity in primal mode
        return self.wo, dr.full(mi.Float, 1, dr.width(wo))


    def scale_factor(self):
        """
        Smooth scale factor that is used to slow down sample movement close
        to geometric discontinuities.
        """

        # The computation is completely independent of any differentiated scene
        # parameters, i.e. everything here is detached.
        with dr.suspend_grad():
            # Initialize some accumulators
            Z = mi.Float(0.0)
            dZ = mi.Vector3f(0.0)
            B = mi.Float(0.0)
            dB_lhs = mi.Vector3f(0.0)
            it = mi.UInt32(0)
            rng = self.rng
            si = self.si
            ray_d = si.to_world(self.wo)
            ray = si.spawn_ray(ray_d)
            ray_frame = mi.Frame3f(ray_d)

            loop = mi.Loop(name="reparameterize_attached_direction(): scale factor",
                           state=lambda: (it, Z, dZ, B, dB_lhs, rng.state))

            # Unroll the entire loop in wavefront mode
            # loop.set_uniform(True) # TODO can we turn this back on? (see self.active in loop condiction)
            loop.set_max_iterations(self.num_rays)
            loop.set_eval_stride(self.num_rays)

            while loop(self.active & (it < self.num_rays)):
                rng_state_backup = rng.state
                sample = mi.Point2f(rng.next_float32(),
                                    rng.next_float32())

                if self.antithetic:
                    repeat = dr.eq(it & 1, 0)
                    rng.state[repeat] = rng_state_backup
                else:
                    repeat = mi.Bool(False)

                # Sample an auxiliary ray from a von Mises Fisher distribution
                omega_local = mi.warp.square_to_von_mises_fisher(sample, self.kappa)

                # Antithetic sampling (optional)
                omega_local.x[repeat] = -omega_local.x
                omega_local.y[repeat] = -omega_local.y

                aux_ray = mi.Ray3f(
                    o = ray.o,
                    d = ray_frame.to_world(omega_local),
                    time = ray.time,
                    wavelengths = ray.wavelengths)

                # Compute an intersection that includes the boundary test
                aux_si = self.scene.ray_intersect(aux_ray,
                                                  ray_flags=mi.RayFlags.All | mi.RayFlags.BoundaryTest,
                                                  coherent=False)

                aux_hit = aux_si.is_valid()

                # Standard boundary test evaluation
                b_test = dr.select(aux_hit, aux_si.boundary_test, 1.0)
                # The horizon of the local hemisphere also counts as an edge here,
                # as e.g. normal maps introduce a discontinuity there.
                horizon = dr.abs(dr.dot(si.n, aux_ray.d))**2
                aux_B = b_test * horizon

                # Augmented boundary test. This quantity will be smoothed as a
                # result of this convolution process.
                B_direct = dr.power(1.0 - aux_B, 3.0)

                # Inverse of vMF density without normalization constant
                inv_vmf_density = dr.rcp(dr.fma(sample.y, dr.exp(-2 * self.kappa), 1 - sample.y))

                # Compute harmonic weight, being wary of division by near-zero values
                w_denom = inv_vmf_density + aux_B - 1
                w_denom_rcp = dr.select(w_denom > 1e-4, dr.rcp(w_denom), 0.0)
                w = dr.power(w_denom_rcp, self.exponent) * inv_vmf_density

                # Analytic weight gradient w.r.t. `ray.d` (detaching inv_vmf_density gradient)
                tmp1 = inv_vmf_density * w * w_denom_rcp * self.kappa * self.exponent
                tmp2 = ray_frame.to_world(mi.Vector3f(omega_local.x, omega_local.y, 0))
                d_w_omega = dr.clamp(tmp1, -1e10, 1e10) * tmp2

                Z += w
                dZ += d_w_omega
                B += w * B_direct
                dB_lhs += d_w_omega * B_direct
                it += 1

            inv_Z = dr.rcp(dr.maximum(Z, 1e-8))
            B = B * inv_Z
            dB = (dB_lhs - dZ * B) * inv_Z

            # Ignore inactive lanes
            B = dr.select(self.active, B, 0.0)
            dB = dr.select(self.active, dB, 0.0)

            return B, dB


    def forward(self):
        """
        Propagate the gradients in the forward direction to 'ray.d' and the
        jacobian determinant 'det'. From a warp field point of view, the
        derivative of 'ray.d' is the warp field direction at 'ray', and
        the derivative of 'det' is the divergence of the warp field at 'ray'.
        """

        # Setup inputs and their gradients
        wo, pdf = mi.Vector3f(self.wo), mi.Float(self.pdf)
        dr.enable_grad([wo, pdf])
        dr.set_grad(wo, self.grad_in('wo'))
        dr.set_grad(pdf, self.grad_in('pdf'))

        # Naïve reparameterization: simply cancel out all movement
        S = wo
        det_S = dr.detach(pdf) / pdf

        if not self.naive:
            # Only scale down movement close to geometric discontinuities
            B, dB = self.scale_factor()
            S_tmp = B * S
            det_S_tmp = dr.dot(dB, self.si.to_world(S)) + B * det_S
            S = S_tmp
            det_S = det_S_tmp

        # Apply reparameterization to direction
        wo_R = dr.normalize(wo - S + dr.detach(S))
        det_R = mi.Float(1.0) - det_S + dr.detach(det_S)

        # Set output gradients
        dr.forward_to(wo_R, det_R)
        self.set_grad_out((dr.select(self.active, dr.grad(wo_R), 0.0),
                           dr.select(self.active, dr.grad(det_R), 0.0)))


    def backward(self):
        # Setup inputs/outputs and their gradients
        grad_dir, grad_det = self.grad_out()
        grad_dir = dr.select(self.active, grad_dir, 0.0)
        grad_det = dr.select(self.active, grad_det, 0.0)

        wo = mi.Vector3f(self.wo)
        pdf = mi.Float(self.pdf)
        dr.enable_grad([wo, pdf])

        # Naïve attached parameterization: simply cancel out all movement
        S = wo
        det_S = dr.detach(pdf) / pdf

        if not self.naive:
            # Only scale down movement close to geometric discontinuities
            B, dB = self.scale_factor()

            S_tmp = B * S
            det_S_tmp = dr.dot(dB, self.si.to_world(S)) + B * det_S
            S = S_tmp; det_S = det_S_tmp
            del S_tmp, det_S_tmp, B, dB

        # Apply reparameterization to direction
        wo_R = dr.normalize(wo - S + dr.detach(S))
        det_R = mi.Float(1.0) - det_S + dr.detach(det_S)

        # Set input gradients
        dr.set_grad(wo_R, grad_dir)
        dr.set_grad(det_R, grad_det)
        dr.enqueue(dr.ADMode.Backward, wo_R, det_R)
        dr.traverse(mi.Float, dr.ADMode.Backward)
        self.set_grad_in('wo', dr.grad(wo))
        self.set_grad_in('pdf', dr.grad(pdf))


    def name(self):
        return "reparameterize_ray()"


def reparameterize_attached_direction(scene: mitsuba.Scene,
                                      rng: mitsuba.PCG32,
                                      wo: mitsuba.Vector3f,
                                      pdf: mitsuba.Float,
                                      si: mitsuba.SurfaceInteraction3f,
                                      num_rays: int=4,
                                      kappa: float=1e5,
                                      exponent: float=3.0,
                                      antithetic: bool=False,
                                      naive: bool=False,
                                      active: mitsuba.Bool = True
) -> Tuple[mitsuba.Vector3f, mitsuba.Float]:
    """
    Reparameterize given ray by "attaching" the derivatives of its direction to
    moving geometry in the scene.

    Parameter ``scene`` (``mitsuba.Scene``):
        Scene containing all shapes.

    Parameter ``rng`` (``mitsuba.PCG32``):
        Random number generator used to sample auxiliary ray directions.

    Parameter ``params`` (``mitsuba.SceneParameters``):
        Scene parameters

    Parameter ``wo`` (``mitsuba.Vector3f``):
        (Attached) sampled direction

    Parameter ``pdf`` (``mitsuba.Float``):
        (Attached) sampling PDF

    Parameter ``si`` (``mitsuba.SurfaceInteraction3f``):
        Current surface interaction

    Parameter ``num_rays`` (``int``):
        Number of auxiliary rays to trace when performing the convolution.

    Parameter ``kappa`` (``float``):
        Kappa parameter of the von Mises Fisher distribution used to sample the
        auxiliary rays.

    Parameter ``exponent`` (``float``):
        Exponent used in the computation of the harmonic weights

    Parameter ``antithetic`` (``bool``):
        Should antithetic sampling be enabled to improve convergence?
        (Default: False)

    Parameter ``naive`` (``bool``):
        Should the naïve version of the reparameterization be used that falls
        back to detached sampling, i.e. cancelling all sample movement?

    Parameter ``active`` (``mitsuba.Bool``):
        Boolean array specifying the active lanes

    Returns → (mitsuba.Vector3f, mitsuba.Float):
        Returns the reparameterized ray direction and the Jacobian
        determinant of the change of variables.
    """

    return dr.custom(_AttachedReparameterizeOp, scene, rng, wo, pdf, si,
                     num_rays, kappa, exponent, antithetic, naive, active)
