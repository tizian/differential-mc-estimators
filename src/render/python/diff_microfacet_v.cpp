#include <mitsuba/render/diff_microfacet.h>
#include <mitsuba/python/python.h>

MI_PY_EXPORT(diff_microfacet) {
    MI_PY_IMPORT_TYPES()

    m.def("lambert_w_zero",
          diff_microfacet::lambert_w_zero<Float>,
          "x"_a, D(diff_microfacet, lambert_w_zero));

    m.def("lambert_w_minus",
          diff_microfacet::lambert_w_minus<Float>,
          "x"_a, D(diff_microfacet, lambert_w_minus));

    m.def("square_to_d_beckmann_neg",
          diff_microfacet::square_to_d_beckmann_neg<Float>,
          "sample"_a, "alpha"_a, D(diff_microfacet, square_to_d_beckmann_neg));

    m.def("square_to_d_beckmann_pos",
          diff_microfacet::square_to_d_beckmann_pos<Float>,
          "sample"_a, "alpha"_a, D(diff_microfacet, square_to_d_beckmann_pos));

    m.def("square_to_d_beckmann_abs",
          diff_microfacet::square_to_d_beckmann_abs<Float>,
          "sample"_a, "alpha"_a, D(diff_microfacet, square_to_d_beckmann_abs));

    m.def("square_to_d_beckmann_neg_pdf",
          diff_microfacet::square_to_d_beckmann_neg_pdf<Float>,
          "m"_a, "alpha"_a, D(diff_microfacet, square_to_d_beckmann_neg_pdf));

    m.def("square_to_d_beckmann_pos_pdf",
          diff_microfacet::square_to_d_beckmann_pos_pdf<Float>,
          "m"_a, "alpha"_a, D(diff_microfacet, square_to_d_beckmann_pos_pdf));

    m.def("square_to_d_beckmann_abs_pdf",
          diff_microfacet::square_to_d_beckmann_abs_pdf<Float>,
          "m"_a, "alpha"_a, D(diff_microfacet, square_to_d_beckmann_abs_pdf));

    m.def("square_to_d_ggx_neg",
          diff_microfacet::square_to_d_ggx_neg<Float>,
          "sample"_a, "alpha"_a, D(diff_microfacet, square_to_d_ggx_neg));

    m.def("square_to_d_ggx_pos",
          diff_microfacet::square_to_d_ggx_pos<Float>,
          "sample"_a, "alpha"_a, D(diff_microfacet, square_to_d_ggx_pos));

    m.def("square_to_d_ggx_abs",
          diff_microfacet::square_to_d_ggx_abs<Float>,
          "sample"_a, "alpha"_a, D(diff_microfacet, square_to_d_ggx_abs));

    m.def("square_to_d_ggx_neg_pdf",
          diff_microfacet::square_to_d_ggx_neg_pdf<Float>,
          "m"_a, "alpha"_a, D(diff_microfacet, square_to_d_ggx_neg_pdf));

    m.def("square_to_d_ggx_pos_pdf",
          diff_microfacet::square_to_d_ggx_pos_pdf<Float>,
          "m"_a, "alpha"_a, D(diff_microfacet, square_to_d_ggx_pos_pdf));

    m.def("square_to_d_ggx_abs_pdf",
          diff_microfacet::square_to_d_ggx_abs_pdf<Float>,
          "m"_a, "alpha"_a, D(diff_microfacet, square_to_d_ggx_abs_pdf));
}
