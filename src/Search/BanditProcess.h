#pragma once
#include <Core/util.h>

namespace rai {
    struct BanditProcess {
        double beta = rai::getParameter<double>("beta"); // discount factor
        double sigma = 0.01; // quantized time unit (in seconds)
        bool empty = true;

        virtual ~BanditProcess() = default;
        
        virtual double compute_gittins_index(int state) const = 0;
        
      };
}