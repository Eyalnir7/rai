#pragma once
#include <Search/ComputeNode.h>
#include <Search/TaskPlan.h>
#include <Search/NodeTypes.h> 
#include <Search/BanditProcess.h>

namespace rai {
  struct Configuration;
}

struct GittinsNode : rai::ComputeNode {
  using ComputeNode::ComputeNode;     
  
  rai::TaskPlan taskPlan = rai::TaskPlan();
  std::unique_ptr<rai::BanditProcess> banditProcess;
  
  // Virtual function to get configuration - returns nullptr by default
  virtual rai::Configuration* getConfiguration() { return nullptr; }

  virtual void initBanditProcess();

  double computePriority() override;

  // bool hasBanditProcess() const { return bandit_process != nullptr; }
};