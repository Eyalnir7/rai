/*  ------------------------------------------------------------------
    Copyright (c) 2011-2024 Marc Toussaint
    email: toussaint@tu-berlin.de

    This code is distributed under the MIT License.
    Please see <root-path>/LICENSE for details.
    --------------------------------------------------------------  */

#include "AStar.h"

#include <fstream>
#include <iomanip>
#include <typeinfo>
#include "../Core/util.h"         // niceTypeidName(...)
#include "Search/ComputeNode.h"          // for ComputeNode::c
#include "../LGP/LGP_computers.h"
#include "Search/GittinsNode.h" // for GittinsNode
#include <Search/NodeTypes.h>

rai::AStar::AStar(const std::shared_ptr<rai::TreeSearchNode>& _root, SearchMode _mode)
  : root(_root), mode(_mode) {
  root->ID = 0;
  mem.append(root);
  addToQueue(root.get());
}

void rai::AStar::step(bool fol) {
  steps++;

  //pop
  TreeSearchNode* node = 0;
  if(mode==astar || mode==FIFO){
    if(!queue.N) {
      LOG(-1) <<"AStar: queue is empty -> failure!";
      return;
    }
    // printFrontier();
    node = queue.pop();
    // downcast the node to a FOL_World_State object
    if (opt.verbose >= 1){
    if(auto folNode = dynamic_cast<FOL_World_State*>(node)){
      str debug;
      folNode->getDecisionSequence(debug);
      // cout << "Exploring node: " << debug << std::endl;
    }}
    // if(mode==astar){
    //   CHECK_GE(node->f_prio, currentLevel, "level needs to increase");
    // }
    currentLevel = node->f_prio;
  }else if(mode==treePolicy){
    node = selectByTreePolicy();
  }
  //    LOG(0) <<"looking at node '" <<*node <<"'";

  //widen
  TreeSearchNode *siblingToBeAdded = 0;
  if(node->needsWidening){
    CHECK(node->parent, "");
    NodeP sibling = node->parent->transition(node->parent->children.N);
    if(sibling){
      CHECK_EQ(sibling->parent, node->parent, "")
      CHECK_GE(sibling->f_prio, currentLevel, "sibling needs to have greater level")
      sibling->ID = mem.N;
      mem.append(sibling);
      siblingToBeAdded = sibling.get();
      //queue.add(sibling->f_prio, sibling.get(), false);
      if(node->parent->getNumDecisions()==-1) sibling->needsWidening=true;
    }
    node->needsWidening=false;
  }

  //compute
  bool wasComplete = node->isComplete;
  if(!node->isComplete){
    if(!fol && opt.verbose>=1) cout << "invested compute in " << *node << "child of" << *node->parent << endl;
    node->compute();
  }

  //depending on state -> drop, reinsert, save as solution, or expand
  if(!node->isFeasible){ //drop node completely

  }else if(!node->isComplete){ //send back to queue
    addToQueue(node);

  // }else if(mode==astar && node->f_prio>currentLevel){ //send back to queue - might not be optimal anymore
  //     addToQueue(node);
  }else if(node->isTerminal){   //save as solution
    // if the node can be casted to a gittinsNode, cast it and print the taskPlan
    // if(auto gittinsNode = dynamic_cast<GittinsNode*>(node)){
    //   std::cout << "GittinsNode found with TaskPlan: " << gittinsNode->taskPlan << std::endl;
    // }
    solutions.append(node);
  }

  else{  //expand or deepen
    CHECK(node->isComplete, "");
    CHECK(!node->isTerminal, "");

    //LOG(0) <<"expanding node '" <<*node <<"'";
    int n = node->getNumDecisions();
    uint createN = n;
    if(n==-1){ createN=1; } //infinity -> add only the first

    for(uint i=0;i<createN;i++) {
      NodeP child = node->transition(i);
      // CHECK_EQ(child->parent, node, "")
          // CHECK_GE(child->f_prio, currentLevel, "children needs to have greater level")
      child->ID = mem.N;
      mem.append(child);
      //child->compute();
      //if(!node->isFeasible) return false;
      addToQueue(child.get());
      if(n==-1) child->needsWidening=true;
    }

  }

  //remember inserting the sibling, FIFO style (also to allow for FIFO mode)
  if(siblingToBeAdded){
    addToQueue(siblingToBeAdded);
  }

  bool becameComplete = (!wasComplete && node->isComplete);
  if(becameComplete && !fol && opt.verbose>=1){
    // printFrontier();
    // convert node to GittinsNode to access taskPlan
    if(auto gittinsNode = dynamic_cast<GittinsNode*>(node)){
      std::cout <<" node '" <<*gittinsNode <<"' became complete and feas: "<< gittinsNode->isFeasible << " after: " << gittinsNode->c <<std::endl;
    }
  }

}

bool rai::AStar::run(int stepsLimit) {
  uint numSol=solutions.N;
  for(;;) {
    step();
    if(solutions.N>numSol) break;
    //      report();
    if(stepsLimit>=0 && (int)steps>=stepsLimit) break;
  }
  if(verbose>0){
    LOG(0) <<"==== DONE ===";
    report();
  }
  if(solutions.N>numSol) return true;
  return false;
}

void rai::AStar::report(){
  std::cout <<" iters: " <<steps
           <<" mem#: " <<mem.N
          <<" queue#: " <<queue.N <<endl;
  if(verbose>2) std::cout <<" queue: " <<queue <<std::endl;
  if(solutions.N) std::cout <<" solutions: " <<solutions.modList();
}

void rai::AStar::printFrontier() const {
  std::cout << "=== Frontier (Queue) Contents ===" << std::endl;
  std::cout << "Queue size: " << queue.N << std::endl;
  
  if(queue.N == 0) {
    std::cout << "Queue is empty." << std::endl;
    return;
  }
  
  // Create a copy of the queue to iterate through without modifying the original
  auto queueCopy = queue;
  
  std::cout << "Priority | Node Name | Additional Info" << std::endl;
  std::cout << "---------|-----------|----------------" << std::endl;
  
  while(queueCopy.N > 0) {
    TreeSearchNode* node = queueCopy.pop();
    
    std::cout << std::setw(8) << std::fixed << std::setprecision(3) << node->f_prio 
              << " | " << std::setw(9) << node->name();
    
    // Check if the node is a GittinsNode and print its TaskPlan
    auto gittinsNode = dynamic_cast<GittinsNode*>(node);
    if(!gittinsNode) std::cout << "not gittins node" << endl;
    if(gittinsNode && !gittinsNode->taskPlan.empty) {
      std::cout << " | parent: " << *gittinsNode->parent << " | compute invested: " << gittinsNode->c;
    } else {
      std::cout << " | (no TaskPlan)";
    }
    
    std::cout << std::endl;
  }
  
  std::cout << "=================================" << std::endl;
}

rai::TreeSearchNode* rai::AStar::selectByTreePolicy(){
  rai::TreeSearchNode *node = root.get();

  //-- TREE POLICY
  while(node->children.N
        //(int)node->children.N == node->getNumDecisions()  //# we are 'inside' the full expanded tree: children for each action -> UCB to select the most promising
        && !node->isTerminal){                       //# we're not at a terminal yet

    // compute the UCB scores for all children
    arr scores(node->children.N);
    for(uint i=0;i<scores.N;i++) scores(i) = node->children(i)->treePolicyScore(i);
    //child->data_Q / child->data_n + beta * sqrt(2. * ::log(node->data_n)/child->data_n);

    // pick the child with highest
    node = node->children(argmax(scores));
  }

  return node;
}

void rai::AStar::addToQueue(TreeSearchNode *node){
  if(mode==FIFO) queue.append(node);
  else queue.add(node->f_prio, node, true);
}
