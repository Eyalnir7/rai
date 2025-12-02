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
  std::unique_ptr<rai::BanditProcess> banditProcess = nullptr; // Pointer to a bandit process, can be null if not used

  // Virtual function to get node type - must be implemented by derived classes
  virtual rai::NodeType getNodeType() const { return rai::NodeType::Other; }
  virtual rai::TaskPlan getTaskPlan() { return taskPlan; }
  
  // Virtual function to get configuration - returns nullptr by default
  virtual rai::Configuration* getConfiguration() { return nullptr; }

  // Convenience function for backward compatibility
  rai::NodeType nodeType() const { return getNodeType(); }

  double computePriority() override;

  // bool hasBanditProcess() const { return bandit_process != nullptr; }
};