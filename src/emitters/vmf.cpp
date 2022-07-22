#include <mitsuba/core/bsphere.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/texture.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class VMFEmitter final : public Emitter<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Emitter, m_flags, m_to_world)
    MI_IMPORT_TYPES(Scene, Shape, Texture)

    VMFEmitter(const Properties &props) : Base(props) {
        /* Until `set_scene` is called, we have no information
           about the scene and default to the unit bounding sphere. */
        m_bsphere = ScalarBoundingSphere3f(ScalarPoint3f(0.f), 1.f);

        if (props.has_property("direction")) {
            if (props.has_property("to_world"))
                Throw("Only one of the parameters 'direction' and 'to_world' "
                      "can be specified at the same time!'");

            ScalarVector3f direction(dr::normalize(props.get<ScalarVector3f>("direction")));
            auto [up, unused] = coordinate_system(direction);

            m_to_world = ScalarTransform4f::look_at(0.0f, ScalarPoint3f(direction), up);
            dr::make_opaque(m_to_world);
        }

        m_intensity = props.texture<Texture>("intensity", Texture::D65(1.f));
        m_kappa = props.get<float>("kappa", 10.f);

        m_flags = +EmitterFlags::Infinite;
    }

    void set_scene(const Scene *scene) override {
        m_bsphere = scene->bbox().bounding_sphere();
        m_bsphere.radius = dr::maximum(math::RayEpsilon<Float>,
                                   m_bsphere.radius * (1.f + math::RayEpsilon<Float>));
    }

    Spectrum eval(const SurfaceInteraction3f &si,
                  Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointEvaluate, active);

        Vector3f v = m_to_world.value().inverse().transform_affine(-si.wi);
        Float scale = warp::square_to_von_mises_fisher_pdf(v, m_kappa);
        Spectrum result = m_intensity->eval(si, active) * scale;
        return depolarizer<Spectrum>(result);
    }

    std::pair<DirectionSample3f, Spectrum>
    sample_direction(const Interaction3f &it,
                     const Point2f &sample,
                     Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSampleDirection, active);

        Vector3f d = warp::square_to_von_mises_fisher(sample, m_kappa);
        Float pdf = warp::square_to_von_mises_fisher_pdf(d, m_kappa);

        d = m_to_world.value().transform_affine(d);

        Float dist = 2.f * m_bsphere.radius;

        DirectionSample3f ds;
        ds.p       = it.p + d * dist;
        ds.n       = -d;
        ds.uv      = Point2f(0.f);
        ds.time    = it.time;
        ds.pdf     = pdf;
        ds.delta   = false;
        ds.emitter = this;
        ds.d       = d;
        ds.dist    = dist;

        SurfaceInteraction3f si = dr::zeros<SurfaceInteraction3f>();
        si.wavelengths = it.wavelengths;

        Spectrum spec = m_intensity->eval(si, active);  // eval(d) / pdf

        return { ds, depolarizer<Spectrum>(spec) };
    }

    Float pdf_direction(const Interaction3f &/*it*/,
                        const DirectionSample3f &ds,
                        Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointEvaluate, active);

        Vector3f d = m_to_world.value().inverse().transform_affine(ds.d);
        return warp::square_to_von_mises_fisher_pdf(d, m_kappa);
    }

    Spectrum eval_direction(const Interaction3f &it,
                            const DirectionSample3f &ds,
                            Mask active) const override {
        Vector3f d = m_to_world.value().inverse().transform_affine(ds.d);
        Float scale = warp::square_to_von_mises_fisher_pdf(d, m_kappa);
        SurfaceInteraction3f si = dr::zeros<SurfaceInteraction3f>();
        si.wavelengths = it.wavelengths;
        Spectrum result = m_intensity->eval(si, active) * scale;
        return depolarizer<Spectrum>(result);
    }

    /// This emitter does not occupy any particular region of space, return an invalid bounding box
    ScalarBoundingBox3f bbox() const override {
        return ScalarBoundingBox3f();
    }

    void traverse(TraversalCallback *callback) override {
        callback->put_object("intensity", m_intensity.get(), +ParamFlags::Differentiable);
        callback->put_parameter("kappa", m_kappa, +ParamFlags::Differentiable);
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "VMFEmitter[" << std::endl
            << "  intensity = " << string::indent(m_intensity) << "," << std::endl
            << "  kappa = " << m_kappa << "," << std::endl
            << "  bsphere = " << m_bsphere << "," << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()
protected:
    ref<Texture> m_intensity;
    ScalarBoundingSphere3f m_bsphere;
    Float m_kappa;
};

MI_IMPLEMENT_CLASS_VARIANT(VMFEmitter, Emitter)
MI_EXPORT_PLUGIN(VMFEmitter, "von Mises-Fisher emitter")
NAMESPACE_END(mitsuba)
