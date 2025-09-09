# pragma once

namespace rai{

  enum class NodeType { RRTNode, LGPPathNode, WaypointsNode, Skeleton, Other }; // Other must be last

  inline const char* toString(NodeType c) {
      switch (c) {
          case NodeType::RRTNode:   return "LGPcomp_RRTpath";
          case NodeType::LGPPathNode: return "LGPcomp_Path";
          case NodeType::WaypointsNode:  return "LGPcomp_Waypoints";
          case NodeType::Skeleton:       return "LGPcomp_Skeleton";
          case NodeType::Other:           return "Other";
      }
  }

  constexpr std::array<NodeType, 3> NODE_TYPE_ORDER = {
      NodeType::WaypointsNode,
      NodeType::RRTNode,
      NodeType::LGPPathNode, 
  };
}