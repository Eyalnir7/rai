/*  ------------------------------------------------------------------
    Copyright (c) 2011-2024 Marc Toussaint
    email: toussaint@tu-berlin.de

    This code is distributed under the MIT License.
    Please see <root-path>/LICENSE for details.
    --------------------------------------------------------------  */

#pragma once

#include "TreeSearchNode.h"
#include "../Algo/priorityQueue.h"

//===========================================================================

namespace rai {

struct AStar_GlobalInfo {
  RAI_PARAM("AStar/", int, verbose, 1)
};

struct AStar {
  enum SearchMode { astar=0, treePolicy=1, FIFO=2, DataExtraction=3 };

  AStar_GlobalInfo opt = AStar_GlobalInfo();

  typedef std::shared_ptr<TreeSearchNode> NodeP;
  rai::Array<NodeP> mem;
  NodeP root;
  PriorityQueue<TreeSearchNode*> queue;
  rai::Array<TreeSearchNode*> solutions;
  uint steps=0;
  int verbose=1;
  double currentLevel=0.;
  SearchMode mode = astar;

  AStar(const std::shared_ptr<TreeSearchNode>& _root, SearchMode _mode = astar);

  void step(bool fol=true);
  void stepGittins();  // Specialized step for GITTINS solver with batched skeleton expansion
  void stepAStar();
  bool run(int stepsLimit=-1);
  void report();
  void printFrontier() const;

  TreeSearchNode* selectByTreePolicy();

private:
  void addToQueue(TreeSearchNode *node);
  
  // GITTINS-specific state for batched skeleton expansion
  uint gittins_maxSkeletons = 10;  // Current batch limit
  std::set<TreeSearchNode*> gittins_triedSkeletons;  // Which skeletons have been tried
};

} //namespace

//===========================================================================


