from __future__ import annotations # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi

from .common import RBIntegrator

class EstimatorComparisonIntegrator(RBIntegrator):
    """
    This integrator compares a variety of Monte Carlo estimators used for
    differential light transport.

    Assumptions:
        1) Direct illumination only, i.e. `max_depth == 2`

        2) Only gradients of BSDF parameters are computed (e.g. surface
           roughness). In particular, gradients from discontinuous geometry
           changes are not supported.
    """

    def __init__(self, props):
        super().__init__(props)

        # Specifies which estimator method should be used. Available options:
        #   Primal mode ...
        #
        #       - "primal_bs"
        #         Primal BSDF sampling

        #       - "primal_es"
        #         Primal emitter sampling

        #       - "primal_mis"
        #         Primal MIS (BSDF + emitter sampling)
        #
        #   Forward and reverse mode diff. mode ...
        #
        #       - "es_detached"
        #         Detached emitter sampling
        #
        #       - "bs_detached"
        #         Detached BSDF sampling
        #
        #       - "bs_attached"
        #         Attached BSDF sampling
        #
        #       - "mis_detached_detached"
        #         Detached MIS weights, detached BSDF sampling, detached emitter sampling
        #
        #       - "mis_attached_attached"
        #         Attached MIS weights, attached BSDF sampling, detached emitter sampling
        #
        #       - "mis_detached_attached"
        #         Detached MIS weights, attached BSDF sampling, detached emitter sampling
        #
        #       - "mis_attached_detached"
        #         Attached MIS weights, detached BSDF sampling, detached emitter sampling
        #
        #       - "bs_detached_diff"
        #         Detached diff. BSDF sampling
        #
        #       - "mis_detached_detached_diff"
        #         Detached MIS weights, detached diff. BSDF sampling, detached emitter sampling
        #
        self.method = props.string('method', 'none')

        # Hide directly visible emitters? (Only relevant in primal modes.)
        self.hide_emitters = props.get('hide_emitters', False)

    def sample(self,
               mode: dr.ADMode,
               scene: mi.Scene,
               sampler: mi.Sampler,
               ray: mi.Ray3f,
               δL: Optional[mi.Spectrum],
               state_in: Optional[mi.Spectrum],
               antithetic_pass: Optional[bool],
               active: mi.Bool,
               **kwargs # Absorbs unused arguments
    ) -> Tuple[mi.Spectrum,
               mi.Bool, mi.Spectrum]:
        """
        See ``ADIntegrator.sample()`` for a description of this interface and
        the role of the various parameters and return values.
        """

        def mis_weight(pdf_a, pdf_b):
            """
            Compute the multiple importance sampling (MIS) weights given the
            densities of two sampling strategies according to the power
            heuristic.
            """
            a2 = dr.sqr(pdf_a)
            return dr.select(pdf_a > 0, a2 / dr.fma(pdf_b, pdf_b, a2), 0)

        # Rendering a primal image? (vs performing forward/reverse-mode AD)
        primal = mode == dr.ADMode.Primal

        # Initialize variables
        ray    = mi.Ray3f(ray)
        L      = mi.Spectrum(0 if primal else state_in)    # Radiance accumulator
        δL     = mi.Spectrum(δL if δL is not None else 0)  # Differential/adjoint radiance
        active = mi.Bool(active)                           # Active SIMD lanes

        # Intersect ray with scene
        si        = scene.ray_intersect(ray)
        valid_ray = active & si.is_valid()
        ctx       = mi.BSDFContext()
        bsdf      = si.bsdf(ray)

        # Account for directly visible emission
        if not self.hide_emitters:
            L += si.emitter(scene).eval(si)

        if primal:
            if self.method == 'primal_es':
                # ------------ Primal emitter sampling ------------
                # -------------------------------------------------

                # Use a emitter sampling strategy
                active_e = active & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)
                ds, emitter_weight = scene.sample_emitter_direction(
                    si, sampler.next_2d(active_e), True, active_e
                )
                active_e &= dr.neq(ds.pdf, 0.0)
                wo = si.to_local(ds.d)

                # Evaluate BSDF
                bsdf_val = bsdf.eval(ctx, si, wo, active_e)
                L = bsdf_val * emitter_weight


            elif self.method == 'primal_bs':
                # ------------ Primal BSDF sampling ------------
                # ----------------------------------------------

                # Use a BSDF sampling strategy
                bs, bsdf_weight = bsdf.sample(
                    ctx, si, sampler.next_1d(active), sampler.next_2d(active), active
                )
                active &= dr.neq(bs.pdf, 0.0)
                ray = si.spawn_ray(si.to_world(bs.wo))
                si_bsdf = scene.ray_intersect(ray, active)

                # Evaluate emitter
                delta = mi.has_flag(bs.sampled_type, mi.BSDFFlags.Delta)
                ds = mi.DirectionSample3f(scene, si_bsdf, si)
                active_b = active & dr.neq(ds.emitter, None) & ~delta
                emitter_val = ds.emitter.eval(si_bsdf, active_b)

                L = bsdf_weight * emitter_val


            else:
                # This is triggered for self.method == 'primal_mis', or any
                # other estimator type when running in primal mode.

                # ------------ MIS between primal BSDF and emitter sampling ------------
                # ----------------------------------------------------------------------

                # First, use an emitter sampling strategy
                active_e = active & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)
                ds, emitter_weight = scene.sample_emitter_direction(
                    si, sampler.next_2d(active_e), True, active_e
                )
                active_e &= dr.neq(ds.pdf, 0.0)
                wo = si.to_local(ds.d)

                # Evaluate BSDF
                bsdf_val, bsdf_pdf = bsdf.eval_pdf(ctx, si, wo, active_e)
                mis = dr.select(ds.delta, 1.0, mis_weight(ds.pdf, bsdf_pdf))

                L += mis * bsdf_val * emitter_weight

                # Second, use a BSDF sampling strategy
                bs, bsdf_weight = bsdf.sample(
                    ctx, si, sampler.next_1d(active), sampler.next_2d(active), active
                )
                active &= dr.neq(bs.pdf, 0.0)
                ray = si.spawn_ray(si.to_world(bs.wo))
                si_bsdf = scene.ray_intersect(ray, active)

                # Evaluate emitter
                delta = mi.has_flag(bs.sampled_type, mi.BSDFFlags.Delta)
                ds = mi.DirectionSample3f(scene, si_bsdf, si)
                active_b = active & dr.neq(ds.emitter, None) & ~delta
                emitter_pdf = scene.pdf_emitter_direction(si, ds, active_b)
                mis = dr.select(active_b, mis_weight(bs.pdf, emitter_pdf), 1.0)
                emitter_val = ds.emitter.eval(si_bsdf, active_b)

                L += mis * bsdf_weight * emitter_val

        else:

            if self.method == 'es_detached':
                # ------------ Detached emitter sampling ------------
                # ---------------------------------------------------

                 # Use an emitter sampling strategy
                 active_e = active & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)
                 ds, emitter_weight = scene.sample_emitter_direction(
                     si, sampler.next_2d(active_e), True, active_e
                 )
                 active_e &= dr.neq(ds.pdf, 0.0)
                 wo = si.to_local(ds.d)

                 with dr.resume_grad(when=not primal):
                     # Only the BSDF evaluation itself is differentiated
                     bsdf_val = bsdf.eval(ctx, si, wo, active_e)

                     L = bsdf_val * emitter_weight


            elif self.method == 'bs_detached':
                # ------------ Detached BSDF sampling ------------
                # ------------------------------------------------

                # Use a BSDF sampling strategy
                bs, _ = bsdf.sample(
                    ctx, si, sampler.next_1d(active), sampler.next_2d(active), active
                )
                active &= dr.neq(bs.pdf, 0.0)
                ray = si.spawn_ray(si.to_world(bs.wo))
                si_bsdf = scene.ray_intersect(ray, active)

                # Evaluate emitter
                delta = mi.has_flag(bs.sampled_type, mi.BSDFFlags.Delta)
                ds = mi.DirectionSample3f(scene, si_bsdf, si)
                active_b = active & dr.neq(ds.emitter, None) & ~delta
                emitter_val = ds.emitter.eval(si_bsdf, active_b)

                with dr.resume_grad(when=not primal):
                    # Only the BSDF evaluation itself is differentiated
                    bsdf_val = bsdf.eval(ctx, si, bs.wo, active)

                    L = bsdf_val / bs.pdf * emitter_val


            elif self.method == 'bs_attached':
                # ------------ Attached BSDF sampling ------------
                # ------------------------------------------------

                with dr.resume_grad(when=not primal):
                    # Use a BSDF sampling strategy, everything is attached here.
                    bs, bsdf_weight = bsdf.sample(
                        ctx, si, sampler.next_1d(active), sampler.next_2d(active), active
                    )
                    active &= dr.neq(bs.pdf, 0.0)
                    ray = si.spawn_ray(si.to_world(bs.wo))
                    si_bsdf = scene.ray_intersect(ray, active)

                    # Evaluate emitter, also being differentiated due to diff. `wo`
                    delta = mi.has_flag(bs.sampled_type, mi.BSDFFlags.Delta)
                    ds = mi.DirectionSample3f(scene, si_bsdf, si)
                    active_b = active & dr.neq(ds.emitter, None) & ~delta
                    emitter_val = ds.emitter.eval(si_bsdf, active_b)

                    L = bsdf_weight * emitter_val


            elif self.method == 'mis_detached_detached':
                # ------------ MIS with detached weights and detached strategies ------------
                #   See Section 3.3.1, Eq. (13)
                # ---------------------------------------------------------------------------

                # First, use an emitter sampling strategy
                active_e = active & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)
                ds, emitter_weight = scene.sample_emitter_direction(
                    si, sampler.next_2d(active_e), True, active_e
                )
                active_e &= dr.neq(ds.pdf, 0.0)
                wo = si.to_local(ds.d)

                with dr.resume_grad(when=not primal):
                    # Only the BSDF evaluation itself is differentiated.
                    # The MIS weights are explicitly detached.
                    bsdf_val, bsdf_pdf = bsdf.eval_pdf(ctx, si, wo, active_e)
                    mis = dr.select(ds.delta, 1.0, mis_weight(ds.pdf, bsdf_pdf))

                    L += dr.detach(mis) * bsdf_val * emitter_weight

                # Second, use a BSDF sampling strategy
                bs, _ = bsdf.sample(
                    ctx, si, sampler.next_1d(active), sampler.next_2d(active), active
                )
                active &= dr.neq(bs.pdf, 0.0)
                ray = si.spawn_ray(si.to_world(bs.wo))
                si_bsdf = scene.ray_intersect(ray, active)

                # Evaluate emitter
                delta = mi.has_flag(bs.sampled_type, mi.BSDFFlags.Delta)
                ds = mi.DirectionSample3f(scene, si_bsdf, si)
                active_b = active & dr.neq(ds.emitter, None) & ~delta
                emitter_pdf = scene.pdf_emitter_direction(si, ds, active_b)
                mis = dr.select(active_b, mis_weight(bs.pdf, emitter_pdf), 1.0)
                emitter_val = ds.emitter.eval(si_bsdf, active_b)

                with dr.resume_grad(when=not primal):
                    # Only the BSDF evaluation itself is differentiated.
                    bsdf_val = bsdf.eval(ctx, si, bs.wo, active)

                    L += mis * bsdf_val / bs.pdf * emitter_val


            elif self.method == 'mis_attached_attached':
                # ------------ MIS with attached weights and attached strategies ------------
                #   See Section 3.3.2, Eq. (14)
                # ---------------------------------------------------------------------------

                with dr.resume_grad(when=not primal):
                    # Both sampling strategies and the MIS weight evaluations
                    # are attached and part of the differentiation.
                    # (Emitter sampling is independent of BSDF parameters and is
                    # technically detached.)

                    # First, use an emitter sampling strategy
                    active_e = active & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)
                    ds, emitter_weight = scene.sample_emitter_direction(
                        si, sampler.next_2d(active_e), True, active_e
                    )
                    active_e &= dr.neq(ds.pdf, 0.0)
                    wo = si.to_local(ds.d)

                    # Evaluate BSDF
                    bsdf_val, bsdf_pdf = bsdf.eval_pdf(ctx, si, wo, active_e)
                    mis = dr.select(ds.delta, 1.0, mis_weight(ds.pdf, bsdf_pdf))

                    L += mis * bsdf_val * emitter_weight

                    # Second, use a BSDF sampling strategy
                    bs, bsdf_weight = bsdf.sample(
                        ctx, si, sampler.next_1d(active), sampler.next_2d(active), active
                    )
                    active &= dr.neq(bs.pdf, 0.0)
                    ray = si.spawn_ray(si.to_world(bs.wo))
                    si_bsdf = scene.ray_intersect(ray, active)

                    # Evaluate emitter
                    delta = mi.has_flag(bs.sampled_type, mi.BSDFFlags.Delta)
                    ds = mi.DirectionSample3f(scene, si_bsdf, si)
                    active_b = active & dr.neq(ds.emitter, None) & ~delta
                    emitter_pdf = scene.pdf_emitter_direction(si, ds, active_b)
                    mis = dr.select(active_b, mis_weight(bs.pdf, emitter_pdf), 1.0)
                    emitter_val = ds.emitter.eval(si_bsdf, active_b)

                    L += mis * bsdf_weight * emitter_val


            elif self.method == 'mis_detached_attached':
                # ------------ MIS with detached weights and attached strategies ------------
                #   See Section 3.3.3, Eq. (15)
                #
                #   WARNING: This is biased!
                # ---------------------------------------------------------------------------

                with dr.resume_grad(when=not primal):
                    # Both sampling strategies are attached, but the MIS weights
                    # are detached.
                    # (Emitter sampling is independent of BSDF parameters and it
                    # technically detached.)

                    # First, use an emitter sampling strategy
                    active_e = active & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)
                    ds, emitter_weight = scene.sample_emitter_direction(
                        si, sampler.next_2d(active_e), True, active_e)
                    active_e &= dr.neq(ds.pdf, 0.0)
                    wo = si.to_local(ds.d)

                    # Evaluate BSDF
                    bsdf_val, bsdf_pdf = bsdf.eval_pdf(ctx, si, wo, active_e)
                    mis = dr.select(ds.delta, 1.0, mis_weight(ds.pdf, bsdf_pdf))

                    L += dr.detach(mis) * bsdf_val * emitter_weight

                    # Second, use a BSDF sampling strategy
                    bs, bsdf_weight = bsdf.sample(
                        ctx, si, sampler.next_1d(active), sampler.next_2d(active), active
                    )
                    active &= dr.neq(bs.pdf, 0.0)
                    ray = si.spawn_ray(si.to_world(bs.wo))
                    si_bsdf = scene.ray_intersect(ray, active)

                    # Evaluate emitter
                    delta = mi.has_flag(bs.sampled_type, mi.BSDFFlags.Delta)
                    ds = mi.DirectionSample3f(scene, si_bsdf, si)
                    active_b = active & dr.neq(ds.emitter, None) & ~delta
                    emitter_pdf = scene.pdf_emitter_direction(si, ds, active_b)
                    mis = dr.select(active_b, mis_weight(bs.pdf, emitter_pdf), 1.0)
                    emitter_val = ds.emitter.eval(si_bsdf, active_b)

                    L += dr.detach(mis) * bsdf_weight * emitter_val


            elif self.method == 'mis_attached_detached':
                # ------------ MIS with attached weights and detached strategies ------------
                #   See Section 3.3.4, Eq. (16)
                #
                #   This is correct, but generally doesn't give any benefits compared
                #   to 'mis_detached_detached'.
                # ---------------------------------------------------------------------------


                # First, use an emitter sampling strategy
                active_e = active & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)
                ds, emitter_weight = scene.sample_emitter_direction(
                    si, sampler.next_2d(active_e), True, active_e
                )
                active_e &= dr.neq(ds.pdf, 0.0)
                wo = si.to_local(ds.d)

                with dr.resume_grad(when=not primal):
                    # Both the BSDF and the MIS weight evaluation are differentiated
                    bsdf_val, bsdf_pdf = bsdf.eval_pdf(ctx, si, wo, active_e)
                    mis = dr.select(ds.delta, 1.0, mis_weight(ds.pdf, bsdf_pdf))

                    L += mis * bsdf_val * emitter_weight

                # Second, use a BSDF sampling strategy
                bs, _ = bsdf.sample(
                    ctx, si, sampler.next_1d(active), sampler.next_2d(active), active
                )
                active &= dr.neq(bs.pdf, 0.0)
                ray = si.spawn_ray(si.to_world(bs.wo))
                si_bsdf = scene.ray_intersect(ray, active)

                # Evaluate emitter
                delta = mi.has_flag(bs.sampled_type, mi.BSDFFlags.Delta)
                ds = mi.DirectionSample3f(scene, si_bsdf, si)
                active_b = active & dr.neq(ds.emitter, None) & ~delta
                emitter_pdf = scene.pdf_emitter_direction(si, ds, active_b)
                emitter_val = ds.emitter.eval(si_bsdf, active_b)

                with dr.resume_grad(when=not primal):
                    # Both the BSDF and the MIS weight evaluation are differentiated.
                    bsdf_val, bsdf_pdf = bsdf.eval_pdf(ctx, si, bs.wo, active)
                    mis = dr.select(active_b, mis_weight(bsdf_pdf, emitter_pdf), 1.0)

                    L += mis * bsdf_val / bs.pdf * emitter_val


            elif self.method == 'bs_detached_diff':
                # ------------ Detached BSDF sampling ---------------------------------------
                #   ... but using the differential microfacet sampling strategy (Section 4.1)
                # ---------------------------------------------------------------------------

                # Select whether we use the positive or negative component for
                # antithetic sampling.
                ctx_diff = mi.BSDFContext()
                if antithetic_pass:
                    ctx_diff.type_mask |= mi.BSDFFlags.DifferentialSamplingPositive
                else:
                    ctx_diff.type_mask |= mi.BSDFFlags.DifferentialSamplingNegative

                # Use a differential BSDF sampling strategy.
                bs, _ = bsdf.sample(
                    ctx_diff, si, sampler.next_1d(active), sampler.next_2d(active), active
                )
                active &= dr.neq(bs.pdf, 0.0)
                ray = si.spawn_ray(si.to_world(bs.wo))
                si_bsdf = scene.ray_intersect(ray, active)

                # Evaluate emitter
                delta = mi.has_flag(bs.sampled_type, mi.BSDFFlags.Delta)
                ds = mi.DirectionSample3f(scene, si_bsdf, si)
                active_b = active & dr.neq(ds.emitter, None) & ~delta
                emitter_val = ds.emitter.eval(si_bsdf, active_b)

                with dr.resume_grad(when=not primal):
                    # Only the BSDF evaluation itself is differentiated
                    bsdf_val = bsdf.eval(ctx, si, bs.wo, active)

                    L = bsdf_val / bs.pdf * emitter_val


            elif self.method == 'mis_detached_detached_diff':
                # ------------ MIS with detached weights and detached strategies ------------
                # ... but using the differential microfacet sampling strategy (Section 4.1)
                # ---------------------------------------------------------------------------

                # Select whether we use the positive or negative component for
                # antithetic sampling.
                ctx_diff = mi.BSDFContext()
                if antithetic_pass:
                    ctx_diff.type_mask |= mi.BSDFFlags.DifferentialSamplingPositive
                else:
                    ctx_diff.type_mask |= mi.BSDFFlags.DifferentialSamplingNegative

                # First, use an emitter sampling strategy
                active_e = active & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)
                ds, emitter_weight = scene.sample_emitter_direction(
                    si, sampler.next_2d(active_e), True, active_e
                )
                active_e &= dr.neq(ds.pdf, 0.0)
                wo = si.to_local(ds.d)

                with dr.resume_grad(when=not primal):
                    # Only the BSDF evaluation itself is differentiated.
                    # The MIS weights are explicitly detached.
                    # The PDF is computed based on the differential sampling strategy.
                    # bsdf_val, bsdf_pdf = bsdf.eval_pdf(ctx_diff, si, wo, active_e)
                    bsdf_pdf = bsdf.pdf(ctx_diff, si, wo, active_e)
                    active_e &= dr.neq(bsdf_pdf, 0.0)
                    bsdf_val = bsdf.eval(ctx, si, wo, active_e)
                    mis = dr.select(ds.delta, 1.0, mis_weight(ds.pdf, bsdf_pdf))

                    L += dr.detach(mis) * bsdf_val * emitter_weight

                # Second, use a differential BSDF sampling strategy
                bs, _ = bsdf.sample(
                    ctx_diff, si, sampler.next_1d(active), sampler.next_2d(active), active
                )
                active &= dr.neq(bs.pdf, 0.0)
                ray = si.spawn_ray(si.to_world(bs.wo))
                si_bsdf = scene.ray_intersect(ray, active)

                # Evaluate emitter
                delta = mi.has_flag(bs.sampled_type, mi.BSDFFlags.Delta)
                ds = mi.DirectionSample3f(scene, si_bsdf, si)
                active_b = active & dr.neq(ds.emitter, None) & ~delta
                emitter_pdf = scene.pdf_emitter_direction(si, ds, active_b)
                mis = dr.select(active_b, mis_weight(bs.pdf, emitter_pdf), 1.0)
                emitter_val = ds.emitter.eval(si_bsdf, active_b)

                with dr.resume_grad(when=not primal):
                    # Only the BSDF evaluation itself is differentiated.
                    bsdf_val = bsdf.eval(ctx, si, bs.wo, active)

                    L += mis * bsdf_val / bs.pdf * emitter_val


        if not primal:
            with dr.resume_grad():
                # Propagate derivatives from/to 'L' based on 'mode'
                if mode == dr.ADMode.Backward:
                    dr.backward_from(δL * L)
                else:
                    δL += dr.forward_to(L)

        return (
            L if primal else δL, # Radiance/differential radiance
            valid_ray,           # Ray validity flag for alpha blending
            L                    # State the for differential phase
        )


    def to_string(self):
        return f'{type(self).__name__}[method = {self.method} ]'

mi.register_integrator("estimator_comparison", lambda props: EstimatorComparisonIntegrator(props))
