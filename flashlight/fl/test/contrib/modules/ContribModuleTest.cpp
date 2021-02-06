/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "flashlight/fl/autograd/autograd.h"
#include "flashlight/fl/common/common.h"
#include "flashlight/fl/contrib/modules/modules.h"
#include "flashlight/fl/nn/nn.h"

using namespace fl;

double timeit(std::function<void()> fn) {
  // warmup
  for (int i = 0; i < 10; ++i) {
    fn();
  }
  af::sync();

  int num_iters = 100;
  af::sync();
  auto start = af::timer::start();
  for (int i = 0; i < num_iters; i++) {
    fn();
  }
  af::sync();
  return af::timer::stop(start) / num_iters;
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
      int batchsize = 6;
      int timesteps = 720;
      int c = 132;
      int nheads = 12;
  auto model = std::make_shared<Transformer>(
      c, c / nheads, c, nheads, timesteps, 0.2, 0.1, false, false);
    
  auto fn = [&]() {

      model->eval();
      auto input = fl::Variable(af::randu(c, timesteps, batchsize), false).as(af::dtype::f16);
      fl::Variable padMask;
      model->forward({input, padMask});
  };
  std::cout << "time took " << timeit(fn) * 1000 << "msec" << std::endl;
  return RUN_ALL_TESTS();
}
