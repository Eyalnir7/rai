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
  std::unique_ptr<rai::BanditProcess> banditProcess = nullptr;
  // int stopping_time = 0;
  
  // Virtual function to get configuration - returns nullptr by default
  virtual rai::Configuration* getConfiguration() { return nullptr; }

  virtual void initBanditProcess();

  virtual rai::TaskPlan getTaskPlan();
  virtual rai::NodeType getNodeType() { return rai::NodeType::Other; }

  // void compute() override;
  double computePriority() override;



  // bool hasBanditProcess() const { return bandit_process != nullptr; }
};