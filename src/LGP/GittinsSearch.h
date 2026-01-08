/*  ------------------------------------------------------------------
    Copyright (c) 2011-2024 Marc Toussaint
    email: toussaint@tu-berlin.de

    This code is distributed under the MIT License.
    Please see <root-path>/LICENSE for details.
    --------------------------------------------------------------  */

#pragma once

#include "../Search/GittinsNode.h"
#include "../Algo/priorityQueue.h"

//===========================================================================

namespace rai {

struct LGPComp2_root;

struct GittinsSearch_GlobalInfo {
  RAI_PARAM("AStar/", int, verbose, 1)
  RAI_PARAM("", int, numTaskPlans, 10)
};

struct GittinsSearch {

  GittinsSearch_GlobalInfo opt = GittinsSearch_GlobalInfo();

  typedef std::shared_ptr<GittinsNode> NodeP;
  rai::Array<NodeP> mem;
  std::shared_ptr<LGPComp2_root> root;
  PriorityQueue<GittinsNode*> queue;
  rai::Array<GittinsNode*> solutions;
  uint steps=0;
  int verbose=1;
  double currentLevel=0.;

  GittinsSearch(const std::shared_ptr<LGPComp2_root>& _root);

  void step();
  bool run(int stepsLimit=-1);
  void report();
  void printFrontier() const;

private:
  void addToQueue(GittinsNode *node);

};

} //namespace

//===========================================================================


