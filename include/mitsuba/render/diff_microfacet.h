#pragma once

#include <mitsuba/core/frame.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/vector.h>

NAMESPACE_BEGIN(mitsuba)

/**
* \brief Implements sampling techniques specific to differentiable
* Microfacet distributions (w.r.t. the roughness parameter.)
*/
NAMESPACE_BEGIN(diff_microfacet)

// =======================================================================
//! @{ \name Lambert W functions used by the diff. Beckmann sampling
// =======================================================================

/// LambertW_0(x)
template <typename Value>
MI_INLINE Value lambert_w_zero(Value x) {
    // Initial guess that will converge to the "0" branch of the LambertW function
    Value w = 0.f;
    // Fixed number of Newton iterations
    for (size_t i = 0; i < 8; ++i) {
        w = (x*dr::exp(-w) + w*w) / (w + 1.f);
    }
    return w;
}

/// LambertW_{-1}(x)
template <typename Value>
MI_INLINE Value lambert_w_minus(Value x) {
    // Initial guess that will converge to the "-1" branch of the LambertW function
    Value w = dr::select(x < -0.1f, -2.f, dr::log(-x) - dr::log(-dr::log(-x)));
    // Fixed number of Newton iterations
    for (size_t i = 0; i < 8; ++i) {
        w = (x*dr::exp(-w) + w*w) / (w + 1.f);
    }
    return w;
}

// =======================================================================
//! @{ \name Sampling functions for the diff. Beckmann distribution
// =======================================================================

/// Sample from the negative part of the dBeckmann distribution
template <typename Value>
MI_INLINE Vector<Value, 3> square_to_d_beckmann_neg(const Point<Value, 2> &sample,
                                                    const Value &alpha) {
    Value alpha_2 = alpha*alpha;

    Value phi       = dr::TwoPi<Value>*sample.x(),
          temp_1    = -lambert_w_zero(-sample.y() * dr::rcp(dr::E<Value>)),
          temp_2    = dr::rsqrt(1 + temp_1*alpha_2),
          tan_theta = dr::sqrt(temp_1)*alpha;

    Value sin_theta = tan_theta * temp_2,
          cos_theta = temp_2;

    auto [sp, cp] = dr::sincos(phi);
    return { sin_theta*cp, sin_theta*sp, cos_theta };
}

/// Sample from the positive part of the dBeckmann distribution
template <typename Value>
MI_INLINE Vector<Value, 3> square_to_d_beckmann_pos(const Point<Value, 2> &sample,
                                                    const Value &alpha) {
    Value alpha_2 = alpha*alpha;

    Value phi       = dr::TwoPi<Value>*sample.x(),
          temp_1    = -lambert_w_minus((sample.y() - 1.f) * dr::rcp(dr::E<Value>)),
          temp_2    = dr::rsqrt(1 + temp_1*alpha_2),
          tan_theta = dr::sqrt(temp_1)*alpha;

    Value sin_theta = tan_theta * temp_2,
          cos_theta = temp_2;

    auto [sp, cp] = dr::sincos(phi);
    return { sin_theta*cp, sin_theta*sp, cos_theta };
}

/// Sample from the combined (absolute valued) dBeckmann distribution
template <typename Value>
MI_INLINE Vector<Value, 3> square_to_d_beckmann_abs(const Point<Value, 2> &sample,
                                                    const Value &alpha) {
    Point<Value, 2> sample_neg = sample,
                    sample_pos = sample;
    sample_neg.y() = 2.f*sample.y();
    sample_pos.y() = 2.f*(sample.y() - 0.5f);

    Vector<Value, 3> m_neg = square_to_d_beckmann_neg(sample_neg, alpha),
                     m_pos = square_to_d_beckmann_pos(sample_pos, alpha);

    return dr::select(sample.y() < 0.5f, m_neg, m_pos);
}

/// Helper for evaluating the dBeckmann distribution
template <typename Value>
MI_INLINE Value square_to_d_beckmann_pdf_aux(const Vector<Value, 3> &m_,
                                             const Value &alpha) {
    using Frame = Frame<Value>;

    Value alpha_2 = alpha*alpha,
          alpha_4 = alpha_2*alpha_2;

    Vector<Value, 3> m = dr::normalize(m_);

    Value ct   = Frame::cos_theta(m),
          tt   = Frame::tan_theta(m),
          ct_2 = ct*ct,
          tt_2 = (m.x()*m.x() + m.y()*m.y()) * dr::rcp(ct_2),
          st   = dr::safe_sqrt(1.f - ct_2);

    Value temp   = 1.f - tt_2 * dr::rcp(alpha_2),
          result = dr::exp(temp) * tt * (tt_2 - alpha_2) / (dr::TwoPi<Value> * alpha_4 * ct_2 * st);
    return result;
}

/// Density of the negative part of the dBeckmann distribution
template <typename Value>
MI_INLINE Value square_to_d_beckmann_neg_pdf(const Vector<Value, 3> &m,
                                             const Value &alpha) {
    using Frame = Frame<Value>;
    Value v = dr::maximum(0.f, -2.f*square_to_d_beckmann_pdf_aux(m, alpha));
    return dr::select(Frame::cos_theta(m) < 1e-9f, 0.f, v);
}

/// Density of the positive part of the dBeckmann distribution
template <typename Value>
MI_INLINE Value square_to_d_beckmann_pos_pdf(const Vector<Value, 3> &m,
                                             const Value &alpha) {
    using Frame = Frame<Value>;
    Value v = dr::maximum(0.f, 2.f*square_to_d_beckmann_pdf_aux(m, alpha));
    return dr::select(Frame::cos_theta(m) < 1e-9f, 0.f, v);
}

/// Density of the positive part of the dBeckmann distribution
template <typename Value>
MI_INLINE Value square_to_d_beckmann_abs_pdf(const Vector<Value, 3> &m,
                                             const Value &alpha) {
    using Frame = Frame<Value>;
    Value v = dr::abs(square_to_d_beckmann_pdf_aux(m, alpha));
    return dr::select(Frame::cos_theta(m) < 1e-9f, 0.f, v);
}

// =======================================================================
//! @{ \name Sampling functions for the diff. GGX distribution
// =======================================================================

/// Sample from the negative part of the dGGX distribution
template <typename Value>
MI_INLINE Vector<Value, 3> square_to_d_ggx_neg(const Point<Value, 2> &sample,
                                               const Value &alpha) {
    Value alpha_2 = alpha*alpha,
          alpha_4 = alpha_2*alpha_2;

    Value phi    = dr::TwoPi<Value>*sample.x(),
          temp_1 = dr::sqrt(1 - sample.y()),
          temp_2 = alpha_2 - 1.f,
          A      = sample.y() * (2.f - 2.f*temp_1 + 2.f*(1 + temp_1)*alpha_4 - sample.y()*temp_2*temp_2),
          B      = dr::rcp(sample.y() + 4.f*temp_1*alpha_2 - sample.y()*alpha_4),
          theta  = 0.5f*dr::atan(2.f*alpha*dr::sqrt(A)*B);

    auto [st, ct] = dr::sincos(theta);
    auto [sp, cp] = dr::sincos(phi);
    return { st*cp, st*sp, ct};
}

/// Sample from the positive part of the dGGX distribution
template <typename Value>
MI_INLINE Vector<Value, 3> square_to_d_ggx_pos(const Point<Value, 2> &sample,
                                               const Value &alpha) {
    Value phi   = dr::TwoPi<Value>*sample.x(),
          C     = alpha*dr::sqrt(-2.f*dr::rcp(dr::sqrt(sample.y()) - 1.f) - 1.f),
          theta = dr::atan(C);

    auto [st, ct] = dr::sincos(theta);
    auto [sp, cp] = dr::sincos(phi);
    return { st*cp, st*sp, ct};
}

/// Sample from the combined (absolute valued) dGGX distribution
template <typename Value>
MI_INLINE Vector<Value, 3> square_to_d_ggx_abs(const Point<Value, 2> &sample,
                                               const Value &alpha) {
    Point<Value, 2> sample_neg = sample,
                    sample_pos = sample;
    sample_neg.y() = 2.f*sample.y();
    sample_pos.y() = 2.f*(sample.y() - 0.5f);

    Vector<Value, 3> m_neg = square_to_d_ggx_neg(sample_neg, alpha),
                     m_pos = square_to_d_ggx_pos(sample_pos, alpha);

    return dr::select(sample.y() < 0.5f, m_neg, m_pos);
}

/// Helper for evaluating the dGGX distribution
template <typename Value>
MI_INLINE Value square_to_d_ggx_pdf_aux(const Vector<Value, 3> &m_,
                                        const Value &alpha) {
    using Frame = Frame<Value>;

    Vector<Value, 3> m = dr::normalize(m_);

    Value ct      = Frame::cos_theta(m),
          ct_2    = ct*ct,
          ct_3    = ct*ct_2,
          tt_2    = (m.x()*m.x() + m.y()*m.y()) * dr::rcp(ct_2),
          alpha_2 = alpha*alpha;

    Value numerator   = 2.f*alpha_2*(tt_2 - alpha_2),
          temp        = alpha_2 + tt_2,
          denominator = dr::Pi<Value>*ct_3*temp*temp*temp,
          result      = numerator * dr::rcp(denominator);

    return result;
}

/// Density of the negative part of the dGGX distribution
template <typename Value>
MI_INLINE Value square_to_d_ggx_neg_pdf(const Vector<Value, 3> &m,
                                        const Value &alpha) {
    using Frame = Frame<Value>;
    Value v = dr::maximum(0.f, -2.f*square_to_d_ggx_pdf_aux(m, alpha));
    return dr::select(Frame::cos_theta(m) < 1e-9f, 0.f, v);
}

/// Density of the positive part of the dGGX distribution
template <typename Value>
MI_INLINE Value square_to_d_ggx_pos_pdf(const Vector<Value, 3> &m,
                                        const Value &alpha) {
    using Frame = Frame<Value>;
    Value v = dr::maximum(0.f, 2.f*square_to_d_ggx_pdf_aux(m, alpha));
    return dr::select(Frame::cos_theta(m) < 1e-9f, 0.f, v);
}

/// Density of the combined (absolute valued) dGGX distribution
template <typename Value>
MI_INLINE Value square_to_d_ggx_abs_pdf(const Vector<Value, 3> &m,
                                        const Value &alpha) {
    using Frame = Frame<Value>;
    Value v = dr::abs(square_to_d_ggx_pdf_aux(m, alpha));
    return dr::select(Frame::cos_theta(m) < 1e-9f, 0.f, v);
}

NAMESPACE_END(diff_microfacet)
NAMESPACE_END(mitsuba)
