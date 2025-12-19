#pragma once
#include <Search/MarkovChain.h>
#include <Core/util.h>
#include <Search/NodeTypes.h>
#include <Search/TaskPlan.h>
#include <utility>

namespace rai {

struct Bandit_GlobalInfo {
  RAI_PARAM("Bandit/", double, beta, 0.9999)
};

struct BanditProcess {
        Bandit_GlobalInfo opt = Bandit_GlobalInfo();
        double beta = opt.beta; // discount factor
        double sigma = 0.01; // quantized time unit (in seconds)
        Array<MarkovChain> markovChains;
        bool empty = true;
        TaskPlan taskPlan;
        NodeType nodeType;

        BanditProcess()
        :opt(),
        beta(opt.beta),
        sigma(0.01),
        empty(true)
        {}

        explicit BanditProcess(const Array<MarkovChain>& chains)
            : opt(), beta(opt.beta), sigma(0.01), markovChains(chains), empty(chains.N == 0) {}

      // Optional move version (recommended)
        explicit BanditProcess(Array<MarkovChain>&& chains)
            : opt(), beta(opt.beta), sigma(0.01), markovChains(std::move(chains)), empty(markovChains.N == 0) {}

        virtual ~BanditProcess() = default;
        
        // Returns pair of (stopping_time, gittins_index)
        virtual std::pair<int, double> compute_gittins_index(double state) const;
        
      };
}