/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include "unit_conv_solver.hpp"

namespace {

auto GetConvTestCases(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    auto type_x = datatype;
    auto type_w = datatype;
    auto type_y = (datatype == miopenInt8) ? miopenInt32 : datatype;

    return std::vector{
        // clang-format off
        TestCase{{1, 8, 8, 8}, {8, 8, 3, 3}, {0, 0}, {1, 1}, {1, 1}, type_x, type_w, type_y},
        TestCase{{1, 3, 16, 16}, {4, 3, 3, 3}, {0, 0}, {1, 1}, {1, 1}, type_x, type_w, type_y},         // Basic convolution
        TestCase{{1, 3, 16, 16}, {4, 3, 5, 5}, {2, 2}, {1, 1}, {1, 1}, type_x, type_w, type_y},         // Stride 2x2 test (downsampling)
        TestCase{{1, 3, 14, 14}, {4, 3, 3, 3}, {2, 2}, {2, 2}, {2, 2}, type_x, type_w, type_y},         // Dilation + padding + stride
        TestCase{{1, 3, 7, 7}, {4, 3, 5, 5}, {0, 0}, {1, 1}, {2, 2}, type_x, type_w, type_y},           // Large dilation on small input
        TestCase{{1, 3, 5, 5}, {4, 3, 3, 3}, {0, 0}, {2, 2}, {1, 1}, type_x, type_w, type_y},           // Stride-only test
        TestCase{{1, 3, 3, 3}, {4, 3, 3, 3}, {0, 0}, {1, 1}, {1, 1}, type_x, type_w, type_y},           // Input same size as kernel
        TestCase{{1, 3, 2, 2}, {4, 3, 3, 3}, {1, 1}, {1, 1}, {1, 1}, type_x, type_w, type_y},           // Input smaller than kernel (should be mostly padded zeros)
        TestCase{{1, 3, 24, 24}, {4, 3, 5, 5}, {3, 3}, {2, 2}, {2, 2}, type_x, type_w, type_y},         // Realistic image size + dilation + padding + stride
        TestCase{{2, 8, 16, 16}, {16, 8, 3, 3}, {1, 1}, {1, 1}, {1, 1}, type_x, type_w, type_y},        // Batched input + increased channels
        TestCase{{1, 16, 64, 64}, {32, 16, 5, 5}, {2, 2}, {1, 1}, {2, 2}, type_x, type_w, type_y},      // Large input
        TestCase{{1, 3, 14, 14}, {4, 3, 3, 3}, {2, 2}, {1, 1}, {3, 3}, type_x, type_w, type_y},         // Large dilation, small input
        TestCase{{1, 8, 32, 32}, {16, 8, 3, 3}, {1, 1}, {4, 4}, {1, 1}, type_x, type_w, type_y},        // Large stride, medium input
        TestCase{{1, 3, 28, 28}, {6, 3, 5, 5}, {2, 2}, {3, 3}, {2, 2}, type_x, type_w, type_y},         // Large stride + dilation
        TestCase{{1, 3, 20, 20}, {4, 3, 3, 3}, {0, 0}, {1, 1}, {5, 5}, type_x, type_w, type_y},         // Very large dilation
        TestCase{{1, 3, 16, 16}, {4, 3, 3, 3}, {0, 0}, {1, 1}, {1, 1}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC},    // same, with NHWC
        TestCase{{1, 3, 16, 16}, {4, 3, 5, 5}, {2, 2}, {1, 1}, {1, 1}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC},
        TestCase{{1, 3, 14, 14}, {4, 3, 3, 3}, {2, 2}, {2, 2}, {2, 2}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC},
        TestCase{{1, 3, 7, 7}, {4, 3, 5, 5}, {0, 0}, {1, 1}, {2, 2}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC},
        TestCase{{1, 3, 5, 5}, {4, 3, 3, 3}, {0, 0}, {2, 2}, {1, 1}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC},
        TestCase{{1, 3, 3, 3}, {4, 3, 3, 3}, {0, 0}, {1, 1}, {1, 1}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC},
        TestCase{{1, 3, 2, 2}, {4, 3, 3, 3}, {1, 1}, {1, 1}, {1, 1}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC},
        TestCase{{1, 3, 24, 24}, {4, 3, 5, 5}, {3, 3}, {2, 2}, {2, 2}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC},
        TestCase{{2, 8, 16, 16}, {16, 8, 3, 3}, {1, 1}, {1, 1}, {1, 1}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC},
        TestCase{{1, 16, 64, 64}, {32, 16, 5, 5}, {2, 2}, {1, 1}, {2, 2}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC},
        TestCase{{1, 3, 14, 14}, {4, 3, 3, 3}, {2, 2}, {1, 1}, {3, 3}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC},
        TestCase{{1, 8, 32, 32}, {16, 8, 3, 3}, {1, 1}, {4, 4}, {1, 1}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC},
        TestCase{{1, 3, 28, 28}, {6, 3, 5, 5}, {2, 2}, {3, 3}, {2, 2}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC},
        TestCase{{1, 3, 20, 20}, {4, 3, 3, 3}, {0, 0}, {1, 1}, {5, 5}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC},
        TestCase{{1, 64, 2, 2}, {64, 64, 2, 2}, {0, 0}, {1, 1}, {1, 1}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC}, // Small spatial, large channels
        TestCase{{1, 4, 5, 5}, {4, 4, 3, 3}, {3, 3}, {1, 1}, {1, 1}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC}, // large padding
        TestCase{{1, 3, 7, 7}, {3, 3, 3, 3}, {1, 2}, {1, 1}, {1, 1}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC}, // Uneven padding (asymmetric)
        TestCase{{1, 3, 7, 7}, {3, 3, 3, 3}, {1, 1}, {2, 1}, {1, 1}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC}, // Uneven stride
        // clang-format on
    };
}

auto GetConvTestCasesFull(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    auto type_x = datatype;
    auto type_w = datatype;
    auto type_y = (datatype == miopenInt8) ? miopenInt32 : datatype;

    return std::vector{
        // clang-format off
        // Regression test for https://github.com/ROCm/MIOpen/issues/2047
        TestCase{{1, 1, 2, 1, 2}, {2, 1, 2, 1, 2}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}, type_x, type_w, type_y},
        // clang-format on
    };
}

const auto& GetTestParams()
{
    static const auto params = [] {
        auto p = miopen::unit_tests::UnitTestConvSolverParams(Gpu::All);
        return p;
    }();
    return params;
}

} // namespace

using GPU_UnitTestConvSolverGemmFwdRestFwd_FP16  = GPU_UnitTestConvSolverFwd_FP16;
using GPU_UnitTestConvSolverGemmFwdRestFwd_BFP16 = GPU_UnitTestConvSolverFwd_BFP16;
using GPU_UnitTestConvSolverGemmFwdRestFwd_FP32  = GPU_UnitTestConvSolverFwd_FP32;
using GPU_UnitTestConvSolverGemmFwdRestFwd_I8    = GPU_UnitTestConvSolverFwd_I8;
using CPU_UnitTestConvSolverGemmFwdRestDevApplicabilityFwd_NONE =
    CPU_UnitTestConvSolverDevApplicabilityFwd_NONE;

TEST_P(GPU_UnitTestConvSolverGemmFwdRestFwd_FP16, GemmFwdRest)
{
    this->RunTest(miopen::solver::conv::GemmFwdRest{});
};

TEST_P(GPU_UnitTestConvSolverGemmFwdRestFwd_BFP16, GemmFwdRest)
{
    this->RunTest(miopen::solver::conv::GemmFwdRest{});
};

TEST_P(GPU_UnitTestConvSolverGemmFwdRestFwd_FP32, GemmFwdRest)
{
    this->RunTest(miopen::solver::conv::GemmFwdRest{});
};

TEST_P(GPU_UnitTestConvSolverGemmFwdRestFwd_I8, GemmFwdRest)
{
    this->RunTest(miopen::solver::conv::GemmFwdRest{});
};

TEST_P(CPU_UnitTestConvSolverGemmFwdRestDevApplicabilityFwd_NONE, GemmFwdRest)
{
    this->RunTest(miopen::solver::conv::GemmFwdRest{});
};

// Smoke tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverGemmFwdRestFwd_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoGEMM),
                                          testing::ValuesIn(GetConvTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverGemmFwdRestFwd_BFP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoGEMM),
                                          testing::ValuesIn(GetConvTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverGemmFwdRestFwd_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoGEMM),
                                          testing::ValuesIn(GetConvTestCases(miopenFloat))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverGemmFwdRestFwd_I8,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoGEMM),
                                          testing::ValuesIn(GetConvTestCases(miopenInt8))));

// Device applicability test
INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverGemmFwdRestDevApplicabilityFwd_NONE,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(GetConvTestCases(miopenFloat)[0])));

// Full tests
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverGemmFwdRestFwd_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoGEMM),
                                          testing::ValuesIn(GetConvTestCasesFull(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverGemmFwdRestFwd_BFP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoGEMM),
                                          testing::ValuesIn(GetConvTestCasesFull(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverGemmFwdRestFwd_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoGEMM),
                                          testing::ValuesIn(GetConvTestCasesFull(miopenFloat))));

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverGemmFwdRestFwd_I8,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoGEMM),
                                          testing::ValuesIn(GetConvTestCasesFull(miopenInt8))));
