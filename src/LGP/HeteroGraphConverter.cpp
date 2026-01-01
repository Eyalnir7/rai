#include <cmath>
#include <algorithm>
#include <Kin/frame.h>
#include <Kin/kin.h>
#include <Core/util.h> 
#include "HeteroGraphConverter.h"
#include "HeteroGraph.h"

rai::String getActionType(rai::String action_name) {
    rai::String action_type;
    if(action_name.contains("pick")) action_type = "pick";
    else action_type = "place";
    return action_type;
}

double getDistance(rai::Frame* f1, rai::Frame* f2) {
    rai::Transformation X1 = f1->ensure_X();
    rai::Transformation X2 = f2->ensure_X();
    
    double dx = X1.pos.x - X2.pos.x;
    double dy = X1.pos.y - X2.pos.y;
    return sqrt(dx*dx + dy*dy);
}

rai::String edgeTypeToString(const rai::String& src, const rai::String& edge, const rai::String& dst) {
    return src + "___" + edge + "___" + dst;
}

torch::Tensor getFeatures(rai::Frame* frame, torch::Device device=torch::kCPU) {
    if (!frame->shape) {
        return torch::zeros({4}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    }
    
    // Get position (first 2 coordinates of absolute position)
    rai::Transformation X = frame->ensure_X();
    std::vector<float> features;
    features.push_back(X.pos.x);
    features.push_back(X.pos.y);
    
    // Get size based on shape type
    const arr& size = frame->shape->size;
    switch (frame->getShapeType()) {
        case rai::ST_ssBox:
        case rai::ST_box:
            if (size.N >= 2) {
                features.push_back(size(0)); // width
                features.push_back(size(1)); // height
            } else {
                features.push_back(0.1); // default size
                features.push_back(0.1);
            }
            break;
        case rai::ST_ssCylinder:
        case rai::ST_cylinder:
            if (size.N >= 1) {
                features.push_back(size(0)); // radius
            } else {
                features.push_back(0.1); // default radius
            }
            // For cylinders, we only need 3 features total (pos_x, pos_y, radius)
            break;
        default:
            features.push_back(0.1);
            features.push_back(0.1);
            break;
    }
    
    return torch::tensor(features, torch::TensorOptions().dtype(torch::kFloat32).device(device));
}

rai::String getNodeType(rai::Frame* frame) {
    if (!frame->getShapeType()) {
        return "unknown";
    }
    
    // Check for logical properties in attributes
    if (frame->ats) {
        // Check if it's marked as an object
        rai::Node *logicObj = frame->ats->findNode("logical");
        if (logicObj) {
            rai::Graph logicGraph = logicObj->graph();
            if (logicGraph.get<bool>("is_object", false)) {
                return "object";
            }
            // Check if it's marked as a place
            if (logicGraph.get<bool>("is_place", false)) {
                return "place_frame";
        }
        }
    }
    
    // Default to shape type
    switch (frame->getShapeType()){
        case rai::ST_ssBox:
            return "ssBox";
        case rai::ST_ssCylinder:
            return "ssCylinder";
        case rai::ST_box:
            return "ssBox"; // Treat regular box as ssBox
        case rai::ST_cylinder:
            return "ssCylinder"; // Treat regular cylinder as ssCylinder
        default:
            return "unknown";
    }
}

std::vector<std::pair<rai::String, rai::String>> getCollisionPairs(const rai::Configuration& C) {
    std::vector<std::pair<rai::String, rai::String>> collision_pairs;
    
    for (rai::Frame* frame : C.frames) {
        if (frame->joint) {
            if (frame->name != "ego") {
                collision_pairs.push_back({frame->name, "ego"});
            }
            for (rai::Frame* frame2 : C.frames) {
                if (frame2 != frame && frame2->name.contains("wall")) {
                    collision_pairs.push_back({frame->name, frame2->name});
                }
            }
        }
    }
    return collision_pairs;
}

bool containsCollisionPair(
    const std::set<std::set<rai::String>>& frozenset_collision_pairs,
    const rai::String& a,
    const rai::String& b)
{
    for (const auto& pair_set : frozenset_collision_pairs) {
        if (pair_set.size() != 2) continue;
        auto it = pair_set.begin();
        const rai::String& p1 = *it++;
        const rai::String& p2 = *it;
        if ((p1 == a && p2 == b) || (p1 == b && p2 == a))
            return true;
    }
    return false;
}

std::vector<rai::Frame*> getCloseObstacles(
    const rai::Configuration& C, 
    const rai::String& object_name,
    const std::vector<std::pair<rai::String, rai::String>>& collision_pairs,
    double threshold
) {
    std::vector<rai::Frame*> close_obstacles;
    
    rai::Frame* object_frame = C.getFrame(object_name, false);
    if (!object_frame) {
        cout << "Object frame " << object_name << " not found in configuration." << endl;
        return close_obstacles;
    }
    
    // Create frozenset equivalent for collision pairs
    std::set<std::set<rai::String>> frozenset_collision_pairs;
    for (const auto& pair : collision_pairs) {
        std::set<rai::String> pair_set = {pair.first, pair.second};
        frozenset_collision_pairs.insert(pair_set);
    }
    
    rai::Transformation object_pos = object_frame->ensure_X();
    
    for (rai::Frame* candidate : C.frames) {
        if (candidate == object_frame || !candidate->shape) continue;
        
        // Check collision pair or goal condition
        std::set<rai::String> check_pair = {object_name, candidate->name};
        bool is_collision_pair = containsCollisionPair(frozenset_collision_pairs, object_name, candidate->name);
        bool is_goal = candidate->name.contains("goal");
        
        if (!is_collision_pair && !is_goal) continue;
        
        // Check distance
        rai::Transformation candidate_pos = candidate->ensure_X();
        double dx = object_pos.pos.x - candidate_pos.pos.x;
        double dy = object_pos.pos.y - candidate_pos.pos.y;
        double distance = sqrt(dx*dx + dy*dy);
        
        if (distance < threshold) {
            close_obstacles.push_back(candidate);
        }
    }
    
    return close_obstacles;
}

IntermediateHeteroData get_hetero_data_input(
    rai::Configuration& C,
    StringAA task_plan,
    torch::Device device,
    int action_number) {
    
    IntermediateHeteroData result;
    
    // Filter task_plan based on action_number (similar to Python implementation)
    StringAA filtered_task_plan;
    if (action_number >= 0) {
        if (action_number == 0) {
            // Only take the first action
            filtered_task_plan.append(task_plan(0));
        } else {
            // Take the previous action and the current action
            filtered_task_plan.append(task_plan(action_number - 1));
            filtered_task_plan.append(task_plan(action_number));
        }
        task_plan = filtered_task_plan;
    }
    
    // Track objects to times mapping (equivalent to objects_to_times in Python)
    std::map<std::string, std::set<int>> objects_to_times;
    
    // Get collision pairs (simplified - in real implementation this would come from configuration)
    std::vector<std::pair<rai::String, rai::String>> collision_pairs = getCollisionPairs(C);
    // For now, we'll generate collision pairs based on proximity during processing
    
    // Helper function to add object to relevant nodes (equivalent to add_object_to_relevant in Python)
    auto add_object_to_relevant = [&](const rai::String& obj_name, int time_step) {
        rai::Frame* frame = C.getFrame(obj_name);
        if (!frame) return;
        
        rai::String node_type = getNodeType(frame);
        torch::Tensor features = getFeatures(frame, device);
        rai::String timestamped_name = obj_name + "_" + STRING(time_step);
        
        // Add to appropriate node type
        if (node_type == "ssBox") {
            result.ssBox_nodes.names.push_back(timestamped_name);
            result.ssBox_nodes.features.push_back(features);
            result.ssBox_nodes.times.push_back(time_step);
        } else if (node_type == "place_frame") {
            result.place_frame_nodes.names.push_back(timestamped_name);
            result.place_frame_nodes.features.push_back(features);
            result.place_frame_nodes.times.push_back(time_step);
        } else if (node_type == "object") {
            result.object_nodes.names.push_back(timestamped_name);
            result.object_nodes.features.push_back(features);
            result.object_nodes.times.push_back(time_step);
        } else if (node_type == "ssCylinder") {
            result.ssCylinder_nodes.names.push_back(timestamped_name);
            result.ssCylinder_nodes.features.push_back(features);
            result.ssCylinder_nodes.times.push_back(time_step);
        }
        
        // Track object times
        std::string obj_name_std = std::string(obj_name.p);;
        if (objects_to_times.find(obj_name_std) == objects_to_times.end()) {
            objects_to_times[obj_name_std] = std::set<int>();
        }
        objects_to_times[obj_name_std].insert(time_step);
    };
    cout << task_plan << endl;
    
    // Process task plan (equivalent to the main loop in Python)
    for (int i = 0; i < task_plan.N; i++) {
        const auto& action = task_plan(i);
        rai::String action_type = getActionType(action(0));
        // Add pick/place node
        rai::String action_node_name = action_type + "_" + STRING(i);
        if (action_type == "pick") {
            result.pick_nodes.names.push_back(action_node_name);
            result.pick_nodes.times.push_back(i);
        } else if (action_type == "place") {
            result.place_nodes.names.push_back(action_node_name);
            result.place_nodes.times.push_back(i);
        }
        
        // Create pick_place_objects list for sink edges
        std::vector<rai::String> pick_place_objects = {action_node_name};
        
        // Add time edge to next action
        if (i < (int)task_plan.N - 1) {
            const auto& next_action = task_plan(i + 1);
            rai::String next_action_node = getActionType(next_action(0)) + "_" + STRING(i+1);
            result.time_edges.edges.push_back({action_node_name, next_action_node});
        }
        
        for(int j=1; j<action.N; j++){
            rai::String object_name = action(j);
            add_object_to_relevant(object_name, i);
            rai::String timestamped_object = object_name + "_" + STRING(i);
            pick_place_objects.push_back(timestamped_object);
            std::vector<rai::Frame*> close_obstacles = getCloseObstacles(C, object_name, collision_pairs, 1.0);
            for (rai::Frame* obstacle : close_obstacles) {
                rai::String obstacle_name = rai::String(obstacle->name);
                add_object_to_relevant(obstacle_name, i);
                rai::String timestamped_obstacle = obstacle_name + "_" + STRING(i);
                result.close_edges.edges.push_back({timestamped_object, timestamped_obstacle});
            }
        }
        
        // Add sink edge
        if (action_type == "pick") {
            result.pick_edges.edges.push_back(pick_place_objects);
        } else if (action_type == "place") {
            result.place_edges.edges.push_back(pick_place_objects);
        }
    }
    
    // Add time edges between object instances at different times
    for (const auto& obj_times_pair : objects_to_times) {
        const std::string& obj_name = obj_times_pair.first;
        std::vector<int> times(obj_times_pair.second.begin(), obj_times_pair.second.end());
        std::sort(times.begin(), times.end());
        
        for (int i = 0; i < (int)times.size() - 1; i++) {
            rai::String current_name = rai::String(obj_name.c_str()) + "_" + STRING(times[i]);
            rai::String next_name = rai::String(obj_name.c_str()) + "_" + STRING(times[i + 1]);
            result.time_edges.edges.push_back({current_name, next_name});
        }
    }
    
    return result;
}

HeteroGraph convertToHeteroGraph(const IntermediateHeteroData& interm) {
    HeteroGraph g;

    // Helper lambda: stack vector<Tensor> into one tensor
    auto stack_or_empty = [](const std::vector<torch::Tensor>& vec) {
        if (vec.empty()) return torch::Tensor();
        return torch::stack(vec);
    };

    // ---- NODE TYPE MAPPING ----
    struct NodeInfo {
        std::string type;
        const IntermediateHeteroData::NodeTypeData* data;
    };

    std::vector<NodeInfo> nodeTypes = {
        {"ssBox",        &interm.ssBox_nodes},
        {"place_frame",  &interm.place_frame_nodes},
        {"object",       &interm.object_nodes},
        {"ssCylinder",   &interm.ssCylinder_nodes},
        {"pick",         &interm.pick_nodes},
        {"place",        &interm.place_nodes}
    };

    // Map node name → (type,index)
    std::unordered_map<std::string, std::pair<std::string,int>> name_to_type_idx;

    // ---- INSERT NODES INTO DICTIONARIES ----
    for (auto& nt : nodeTypes) {
        const auto& names = nt.data->names;
        size_t num_nodes = names.size();
        
        // Sanity check for corrupted size
        if (num_nodes > 1000000) {
            std::cerr << "ERROR: Suspiciously large number of nodes (" << num_nodes 
                      << ") for type " << nt.type << std::endl;
            continue;
        }
        
        // Debug: Check times vector size matches nodes
        if (nt.data->times.size() != num_nodes) {
            std::cerr << "ERROR: Mismatch for " << nt.type << " - names.size()=" 
                      << num_nodes << " but times.size()=" << nt.data->times.size() << std::endl;
            continue;
        }

        // record mapping for later edge construction
        for (int i = 0; i < (int)num_nodes; i++)
            name_to_type_idx[std::string(names[i])] = {nt.type, i};

        // fill features (only for feature node types, not constraint types pick/place)
        if (nt.type != "pick" && nt.type != "place") {
            torch::Tensor features = stack_or_empty(nt.data->features);
            if (features.defined() && features.numel() > 0) {
                g.x_dict[nt.type] = features;
            }
        }

        // fill times - but only if we have nodes
        if (num_nodes > 0) {
            torch::Tensor t = torch::tensor(nt.data->times, torch::kLong);
            g.times_dict[nt.type] = t;
        }

        // batch indices: all zeros for single graph (not batched)
        g.batch_dict[nt.type] = torch::zeros({(long)num_nodes}, torch::kLong);
    }

    // ---- BUILD EDGE INDEX DICTIONARY ----

    auto add_edge_to_dict = [&](const std::string& srcType,
                                const std::string& edgeName,
                                const std::string& dstType,
                                const std::vector<int>& srcIdxs,
                                const std::vector<int>& dstIdxs)
    {
        auto edge_index = torch::stack({
            torch::tensor(srcIdxs, torch::kLong),
            torch::tensor(dstIdxs, torch::kLong)
        });

        std::string key = srcType + "___" + edgeName + "___" + dstType;
        g.edge_index_dict[key] = edge_index;
    };


    auto process_pair_edges = [&](const std::string& edgeName,
                                  const EdgeData& ed)
    {
        // Accumulate per (srcType,dstType) groups
        std::unordered_map<std::string, std::vector<int>> src_acc;
        std::unordered_map<std::string, std::vector<int>> dst_acc;

        for (auto& e : ed.edges) {
            auto it1 = name_to_type_idx.find(std::string(e.first));
            auto it2 = name_to_type_idx.find(std::string(e.second));
            if (it1 == name_to_type_idx.end() ||
                it2 == name_to_type_idx.end()) continue;

            auto [t1, i1] = it1->second;
            auto [t2, i2] = it2->second;

            std::string key = t1 + "___" + t2;
            src_acc[key].push_back(i1);
            dst_acc[key].push_back(i2);

            if (!ed.directed) {
                // add reverse edge
                std::string keyR = t2 + "___" + t1;
                src_acc[keyR].push_back(i2);
                dst_acc[keyR].push_back(i1);
            }
        }

        // Now build tensors and insert
        for (auto& kv : src_acc) {
            std::string types = kv.first;
            auto pos = types.find("___");
            std::string srcT = types.substr(0, pos);
            std::string dstT = types.substr(pos + 3);

            add_edge_to_dict(srcT, edgeName, dstT, kv.second, dst_acc[types]);
        }
    };

    // --- process simple pair edges ---
    process_pair_edges("close_edge", interm.close_edges);
    process_pair_edges("time_edge",  interm.time_edges);

    // --- process sink edges ---
    auto process_sink_edges = [&](const std::string& edgeName,
                                  const SinkEdgeData& sed)
    {
        std::unordered_map<std::string, std::vector<int>> src_acc;
        std::unordered_map<std::string, std::vector<int>> dst_acc;

        for (auto& edgeList : sed.edges) {
            if (edgeList.size() < 2) continue;

            std::string sinkName = std::string(edgeList[0]);
            auto itSink = name_to_type_idx.find(sinkName);
            if (itSink == name_to_type_idx.end()) continue;

            auto [sinkType, sinkIdx] = itSink->second;

            for (int k = 1; k < (int)edgeList.size(); k++) {
                auto itC = name_to_type_idx.find(std::string(edgeList[k]));
                if (itC == name_to_type_idx.end()) continue;

                auto [cType, cIdx] = itC->second;

                std::string key = cType + "___" + sinkType;
                src_acc[key].push_back(cIdx);
                dst_acc[key].push_back(sinkIdx);

                if (!sed.directed) {
                    std::string keyR = sinkType + "___" + cType;
                    src_acc[keyR].push_back(sinkIdx);
                    dst_acc[keyR].push_back(cIdx);
                }
            }
        }

        for (auto& kv : src_acc) {
            auto types = kv.first;
            auto pos = types.find("___");
            std::string srcT = types.substr(0, pos);
            std::string dstT = types.substr(pos + 3);

            add_edge_to_dict(srcT, edgeName, dstT, kv.second, dst_acc[types]);
        }
    };

    process_sink_edges("pick_edge",  interm.pick_edges);
    process_sink_edges("place_edge", interm.place_edges);

    return g;
}

