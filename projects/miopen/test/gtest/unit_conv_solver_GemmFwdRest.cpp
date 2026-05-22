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

     std::vector<TestCase> test_cases;
    test_cases.emplace_back(TestCase{{1, 8, 8, 8, 1}, {8, 8, 3, 3, 1}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}, type_x, type_w, type_y});
    test_cases.emplace_back(TestCase{{1, 8, 8, 8}, {8, 8, 3, 3}, {0, 0}, {1, 1}, {1, 1}, type_x, type_w, type_y});
    test_cases.emplace_back(TestCase{{1, 3, 16, 16}, {4, 3, 3, 3}, {0, 0}, {1, 1}, {1, 1}, type_x, type_w, type_y});         // Basic convolution
    test_cases.emplace_back(TestCase{{1, 3, 16, 16}, {4, 3, 5, 5}, {2, 2}, {1, 1}, {1, 1}, type_x, type_w, type_y});         // Stride 2x2 test (downsampling)
    test_cases.emplace_back(TestCase{{1, 3, 14, 14}, {4, 3, 3, 3}, {2, 2}, {2, 2}, {2, 2}, type_x, type_w, type_y});         // Dilation + padding + stride
    test_cases.emplace_back(TestCase{{1, 3, 7, 7}, {4, 3, 5, 5}, {0, 0}, {1, 1}, {2, 2}, type_x, type_w, type_y});           // Large dilation on small input
    test_cases.emplace_back(TestCase{{1, 3, 5, 5}, {4, 3, 3, 3}, {0, 0}, {2, 2}, {1, 1}, type_x, type_w, type_y});           // Stride-only test
    test_cases.emplace_back(TestCase{{1, 3, 3, 3}, {4, 3, 3, 3}, {0, 0}, {1, 1}, {1, 1}, type_x, type_w, type_y});           // Input same size as kernel
    test_cases.emplace_back(TestCase{{1, 3, 2, 2}, {4, 3, 3, 3}, {1, 1}, {1, 1}, {1, 1}, type_x, type_w, type_y});           // Input smaller than kernel (should be mostly padded zeros)
    test_cases.emplace_back(TestCase{{1, 3, 24, 24}, {4, 3, 5, 5}, {3, 3}, {2, 2}, {2, 2}, type_x, type_w, type_y});         // Realistic image size + dilation + padding + stride
    test_cases.emplace_back(TestCase{{2, 8, 16, 16}, {16, 8, 3, 3}, {1, 1}, {1, 1}, {1, 1}, type_x, type_w, type_y});        // Batched input + increased channels
    test_cases.emplace_back(TestCase{{1, 16, 64, 64}, {32, 16, 5, 5}, {2, 2}, {1, 1}, {2, 2}, type_x, type_w, type_y});      // Large input
    test_cases.emplace_back(TestCase{{1, 3, 14, 14}, {4, 3, 3, 3}, {2, 2}, {1, 1}, {3, 3}, type_x, type_w, type_y});         // Large dilation, small input
    test_cases.emplace_back(TestCase{{1, 3, 28, 28}, {6, 3, 5, 5}, {2, 2}, {3, 3}, {2, 2}, type_x, type_w, type_y});         // Large stride + dilation
    test_cases.emplace_back(TestCase{{1, 3, 20, 20}, {4, 3, 3, 3}, {0, 0}, {1, 1}, {5, 5}, type_x, type_w, type_y});         // Very large dilation
    test_cases.emplace_back(TestCase{{1, 8, 32, 32}, {16, 8, 3, 3}, {1, 1}, {4, 4}, {1, 1}, type_x, type_w, type_y});        // Large stride, medium input
    test_cases.emplace_back(TestCase{{1, 4, 64, 64}, {32, 4, 5, 5}, {1, 1}, {1, 1}, {1, 1}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC});
    test_cases.emplace_back(TestCase{{1, 4, 5, 5}, {4, 4, 3, 3}, {1, 1}, {1, 1}, {1, 1}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC});    
    test_cases.emplace_back(TestCase{{1, 4, 5, 5}, {4, 4, 3, 3}, {1, 1}, {1, 1}, {1, 1}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC});    
    test_cases.emplace_back(TestCase{{1, 4, 5, 5}, {4, 4, 3, 3}, {0, 0}, {1, 1}, {1, 1}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC}); // large padding 30
    test_cases.emplace_back(TestCase{{1, 2, 5, 5}, {4, 2, 3, 3}, {0, 0}, {1, 1}, {1, 1}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC}); // large padding 30

    test_cases.emplace_back(TestCase{{1, 3, 16, 16}, {4, 3, 3, 3}, {0, 0}, {1, 1}, {1, 1}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC});    // same, with NHWC
    test_cases.emplace_back(TestCase{{1, 3, 16, 16}, {4, 3, 5, 5}, {2, 2}, {1, 1}, {1, 1}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC});
    test_cases.emplace_back(TestCase{{1, 3, 14, 14}, {4, 3, 3, 3}, {2, 2}, {2, 2}, {2, 2}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC});
    test_cases.emplace_back(TestCase{{1, 3, 7, 7}, {4, 3, 5, 5}, {0, 0}, {1, 1}, {2, 2}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC});
    test_cases.emplace_back(TestCase{{1, 3, 5, 5}, {4, 3, 3, 3}, {0, 0}, {2, 2}, {1, 1}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC});
    test_cases.emplace_back(TestCase{{1, 3, 3, 3}, {4, 3, 3, 3}, {0, 0}, {1, 1}, {1, 1}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC});
    test_cases.emplace_back(TestCase{{1, 3, 2, 2}, {4, 3, 3, 3}, {1, 1}, {1, 1}, {1, 1}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC});
    test_cases.emplace_back(TestCase{{1, 3, 24, 24}, {4, 3, 5, 5}, {3, 3}, {2, 2}, {2, 2}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC});
    test_cases.emplace_back(TestCase{{2, 8, 16, 16}, {16, 8, 3, 3}, {1, 1}, {1, 1}, {1, 1}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC});
    test_cases.emplace_back(TestCase{{1, 3, 14, 14}, {4, 3, 3, 3}, {2, 2}, {1, 1}, {3, 3}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC});
    test_cases.emplace_back(TestCase{{1, 8, 32, 32}, {16, 8, 3, 3}, {1, 1}, {4, 4}, {1, 1}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC});
    test_cases.emplace_back(TestCase{{1, 3, 28, 28}, {6, 3, 5, 5}, {2, 2}, {3, 3}, {2, 2}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC});
    test_cases.emplace_back(TestCase{{1, 3, 20, 20}, {4, 3, 3, 3}, {0, 0}, {1, 1}, {5, 5}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC});
    test_cases.emplace_back(TestCase{{1, 64, 2, 2}, {64, 64, 2, 2}, {0, 0}, {1, 1}, {1, 1}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC}); // Small spatial, large channels 29
    test_cases.emplace_back(TestCase{{1, 3, 7, 7}, {3, 3, 3, 3}, {1, 2}, {1, 1}, {1, 1}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC}); // Uneven padding (asymmetric) 31
    test_cases.emplace_back(TestCase{{1, 3, 7, 7}, {3, 3, 3, 3}, {1, 1}, {2, 1}, {1, 1}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC}); // Un

    // NDHWC tests
    test_cases.emplace_back(TestCase{{2, 16, 5, 5, 5}, {32, 16, 1, 1, 1}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}, type_x, type_w, type_y, miopenTensorNDHWC, miopenTensorNDHWC});
    test_cases.emplace_back(TestCase{{1, 64, 7, 7, 7}, {16, 64, 1, 1, 1}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}, type_x, type_w, type_y, miopenTensorNDHWC, miopenTensorNDHWC});
    test_cases.emplace_back(TestCase{{1, 8, 14, 14, 14}, {16, 8, 3, 3, 3}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}, type_x, type_w, type_y, miopenTensorNDHWC, miopenTensorNDHWC});
    test_cases.emplace_back(TestCase{{4, 3, 10, 10, 10}, {8, 3, 3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, type_x, type_w, type_y, miopenTensorNDHWC, miopenTensorNDHWC});
    test_cases.emplace_back(TestCase{{2, 4, 16, 16, 16}, {8, 4, 3, 3, 3}, {1, 1, 1}, {2, 2, 2}, {1, 1, 1}, type_x, type_w, type_y, miopenTensorNDHWC, miopenTensorNDHWC});
    test_cases.emplace_back(TestCase{{1, 3, 15, 30, 30}, {16, 3, 5, 7, 7}, {2, 3, 3}, {3, 4, 4}, {1, 1, 1}, type_x, type_w, type_y, miopenTensorNDHWC, miopenTensorNDHWC});
    test_cases.emplace_back(TestCase{{1, 8, 15, 15, 15}, {8, 8, 3, 3, 3}, {2, 2, 2}, {1, 1, 1}, {2, 2, 2}, type_x, type_w, type_y, miopenTensorNDHWC, miopenTensorNDHWC});
    test_cases.emplace_back(TestCase{{2, 4, 20, 20, 20}, {4, 4, 3, 3, 3}, {3, 3, 3}, {1, 1, 1}, {3, 3, 3}, type_x, type_w, type_y, miopenTensorNDHWC, miopenTensorNDHWC});
    test_cases.emplace_back(TestCase{{2, 3, 8, 16, 16}, {16, 3, 1, 3, 3}, {0, 1, 1}, {1, 1, 1}, {1, 1, 1}, type_x, type_w, type_y, miopenTensorNDHWC, miopenTensorNDHWC});
    test_cases.emplace_back(TestCase{{1, 16, 10, 8, 8}, {16, 16, 3, 1, 1}, {1, 0, 0}, {1, 1, 1}, {1, 1, 1}, type_x, type_w, type_y, miopenTensorNDHWC, miopenTensorNDHWC});
    test_cases.emplace_back(TestCase{{1, 4, 7, 9, 11}, {8, 4, 3, 2, 4}, {0, 1, 0}, {1, 2, 3}, {2, 1, 1}, type_x, type_w, type_y, miopenTensorNDHWC, miopenTensorNDHWC});


    if(datatype != miopenInt8) {
    test_cases.emplace_back(TestCase{{1, 4, 16, 16}, {4, 2, 3, 3}, {0, 0}, {1, 1}, {1, 1}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC,2}); 
    test_cases.emplace_back(TestCase{{1, 4, 5, 5}, {4, 2, 3, 3}, {0, 0}, {1, 1}, {1, 1}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC,2}); 
    test_cases.emplace_back(TestCase{{1, 2, 3, 3}, {2, 1, 2, 2}, {0, 0}, {1, 1}, {1, 1}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC,2});  
    test_cases.emplace_back(TestCase{{1, 4, 3, 3}, {2, 2, 2, 2}, {0, 0}, {1, 1}, {1, 1}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC,2});  
    test_cases.emplace_back(TestCase{{1, 4, 3, 3}, {4, 2, 2, 2}, {0, 0}, {1, 1}, {1, 1}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC,2});  
    test_cases.emplace_back(TestCase{{1, 4, 3, 3}, {4, 2, 2, 2}, {0, 0}, {1, 1}, {1, 1}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC,2}); 
    test_cases.emplace_back(TestCase{{1, 3, 7, 7}, {12, 1, 5, 5}, {0, 0}, {1, 1}, {2, 2}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC,3});
    test_cases.emplace_back(TestCase{{1, 3, 5, 5}, {9, 1, 3, 3}, {0, 0}, {2, 2}, {1, 1}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC,3});
    test_cases.emplace_back(TestCase{{1, 3, 3, 3}, {6, 1, 3, 3}, {0, 0}, {1, 1}, {1, 1}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC,3});
    test_cases.emplace_back(TestCase{{1, 3, 2, 2}, {3, 1, 3, 3}, {1, 1}, {1, 1}, {1, 1}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC,3});
    test_cases.emplace_back(TestCase{{1, 3, 24, 24}, {3, 1, 5, 5}, {3, 3}, {2, 2}, {2, 2}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC,3});
    test_cases.emplace_back(TestCase{{2, 8, 16, 16}, {16, 2, 3, 3}, {1, 1}, {1, 1}, {1, 1}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC,4});
    test_cases.emplace_back(TestCase{{2, 8, 16, 16}, {16, 1, 3, 3}, {1, 1}, {1, 1}, {1, 1}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC,8});
    test_cases.emplace_back(TestCase{{1, 16, 64, 64}, {32, 1, 5, 5}, {2, 2}, {1, 1}, {2, 2}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC,16});
    test_cases.emplace_back(TestCase{{1, 16, 64, 64}, {32, 2, 5, 5}, {2, 2}, {1, 1}, {2, 2}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC,8});
    test_cases.emplace_back(TestCase{{1, 4, 14, 14}, {4, 2, 3, 3}, {2, 2}, {1, 1}, {3, 3}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC,2});
    test_cases.emplace_back(TestCase{{1, 8, 32, 32}, {16, 4, 3, 3}, {1, 1}, {4, 4}, {1, 1}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC,2});
    test_cases.emplace_back(TestCase{{1, 4, 28, 28}, {6, 2, 5, 5}, {2, 2}, {3, 3}, {2, 2}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC,2});
    test_cases.emplace_back(TestCase{{1, 4, 20, 20}, {4, 2, 3, 3}, {0, 0}, {1, 1}, {5, 5}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC,2});
    test_cases.emplace_back(TestCase{{1, 64, 2, 2}, {64, 4, 2, 2}, {0, 0}, {1, 1}, {1, 1}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC,16}); // Small spatial, large channels 29
    test_cases.emplace_back(TestCase{{1, 64, 2, 2}, {64, 32, 2, 2}, {0, 0}, {1, 1}, {1, 1}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC,2}); // Small spatial, large channels 29
    test_cases.emplace_back(TestCase{{1, 4, 5, 5}, {4, 2, 3, 3}, {3, 3}, {1, 1}, {1, 1}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC,2}); // large padding 30
    test_cases.emplace_back(TestCase{{1, 4, 7, 7}, {6, 2, 3, 3}, {1, 2}, {1, 1}, {1, 1}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC,2}); // Uneven padding (asymmetric) 31
    test_cases.emplace_back(TestCase{{1, 4, 7, 7}, {8, 2, 3, 3}, {1, 1}, {2, 1}, {1, 1}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC,2}); // Un    TestCase{{1, 4, 16, 16}, {4, 2, 3, 3}, {0, 0}, {1, 1}, {1, 1}, type_x, type_w, type_y, miopenTensorNCHW,miopenTensorNCHW, 2}, 
    test_cases.emplace_back(TestCase{{1, 4, 5, 5}, {4, 2, 3, 3}, {0, 0}, {1, 1}, {1, 1}, type_x, type_w, type_y, miopenTensorNHWC,miopenTensorNHWC,2});
    }
    return test_cases;
}

auto GetConvTestCasesFull(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    auto type_x = datatype;
    auto type_w = datatype;
    auto type_y = (datatype == miopenInt8) ? miopenInt32 : datatype;

    auto smoke_test_cases = GetConvTestCases(datatype);
    smoke_test_cases.push_back( TestCase{{1, 1, 2, 1, 2}, {2, 1, 2, 1, 2}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}, type_x, type_w, type_y});
    return smoke_test_cases;
    // return std::vector{
    //     // clang-format off
    //     // Regression test for https://github.com/ROCm/MIOpen/issues/2047
    //     TestCase{{1, 1, 2, 1, 2}, {2, 1, 2, 1, 2}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}, type_x, type_w, type_y},
    //     // clang-format on
    // };
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
