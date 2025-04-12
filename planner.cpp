/*=================================================================
 *
 * planner.cpp
 *
 *=================================================================*/
#include <math.h>
#include <random>
#include <vector>
#include <array>
#include <algorithm>

#include <tuple>
#include <string>
#include <stdexcept>
#include <regex> // For regex and split logic
#include <iostream> // cout, endl
#include <fstream> // For reading/writing files
#include <assert.h> 
// #include <flann/flann.hpp>
#include <queue> // For priority queue in A* search
#include <unordered_map> // For storing graph nodes
#include <limits> // For setting infinity distance
#include <memory>
#include <chrono>


/* Input Arguments */
#define	MAP_IN      prhs[0]
#define	ARMSTART_IN	prhs[1]
#define	ARMGOAL_IN     prhs[2]
#define	PLANNER_ID_IN     prhs[3]

/* Planner Ids */
#define RRT         0
#define RRTCONNECT  1
#define RRTSTAR     2
#define PRM         3

/* Output Arguments */
#define	PLAN_OUT	plhs[0]
#define	PLANLENGTH_OUT	plhs[1]

#define GETMAPINDEX(X, Y, XSIZE, YSIZE) (Y*XSIZE + X)

#if !defined(MAX)
#define	MAX(A, B)	((A) > (B) ? (A) : (B))
#endif

#if !defined(MIN)
#define	MIN(A, B)	((A) < (B) ? (A) : (B))
#endif

#define PI 3.141592654

//the length of each link in the arm
#define LINKLENGTH_CELLS 10

#ifndef MAPS_DIR
#define MAPS_DIR "../maps"
#endif
#ifndef OUTPUT_DIR
#define OUTPUT_DIR "../output"
#endif


// Some potentially helpful imports
using std::vector;
using std::array;
using std::string;
using std::runtime_error;
using std::tuple;
using std::make_tuple;
using std::tie;
using std::cout;
using std::endl;
using std::pair;
using std::copy;

//*******************************************************************************************************************//
//                                                                                                                   //
//                                                GIVEN FUNCTIONS                                                    //
//                                                                                                                   //
//*******************************************************************************************************************//

/// @brief 
/// @param filepath 
/// @return map, x_size, y_size
tuple<double*, int, int> loadMap(string filepath) {
	std::FILE *f = fopen(filepath.c_str(), "r");
	if (f) {
	}
	else {
		printf("Opening file failed! \n");
		throw runtime_error("Opening map file failed!");
	}
	int height, width;
	if (fscanf(f, "height %d\nwidth %d\n", &height, &width) != 2) {
		throw runtime_error("Invalid loadMap parsing map metadata");
	}
	
	////// Go through file and add to m_occupancy
	double* map = new double[height*width];

	double cx, cy, cz;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			char c;
			do {
				if (fscanf(f, "%c", &c) != 1) {
					throw runtime_error("Invalid parsing individual map data");
				}
			} while (isspace(c));
			if (!(c == '0')) { 
				map[y+x*width] = 1; // Note transposed from visual
			} else {
				map[y+x*width] = 0;
			}
		}
	}
	fclose(f);
	return make_tuple(map, width, height);
}

// Splits string based on deliminator
vector<string> split(const string& str, const string& delim) {   
		// https://stackoverflow.com/questions/14265581/parse-split-a-string-in-c-using-string-delimiter-standard-c/64886763#64886763
		const std::regex ws_re(delim);
		return { std::sregex_token_iterator(str.begin(), str.end(), ws_re, -1), std::sregex_token_iterator() };
}


double* doubleArrayFromString(string str) {
	vector<string> vals = split(str, ",");
	double* ans = new double[vals.size()];
	for (int i = 0; i < vals.size(); ++i) {
		ans[i] = std::stod(vals[i]);
	}
	return ans;
}

bool equalDoubleArrays(double* v1, double *v2, int size) {
    for (int i = 0; i < size; ++i) {
        if (abs(v1[i]-v2[i]) > 1e-3) {
            cout << endl;
            return false;
        }
    }
    return true;
}

typedef struct {
	int X1, Y1;
	int X2, Y2;
	int Increment;
	int UsingYIndex;
	int DeltaX, DeltaY;
	int DTerm;
	int IncrE, IncrNE;
	int XIndex, YIndex;
	int Flipped;
} bresenham_param_t;

void ContXY2Cell(double x, double y, short unsigned int* pX, short unsigned int *pY, int x_size, int y_size) {
	double cellsize = 1.0;
	//take the nearest cell
	*pX = (int)(x/(double)(cellsize));
	if( x < 0) *pX = 0;
	if( *pX >= x_size) *pX = x_size-1;

	*pY = (int)(y/(double)(cellsize));
	if( y < 0) *pY = 0;
	if( *pY >= y_size) *pY = y_size-1;
}


void get_bresenham_parameters(int p1x, int p1y, int p2x, int p2y, bresenham_param_t *params) {
	params->UsingYIndex = 0;

	if (fabs((double)(p2y-p1y)/(double)(p2x-p1x)) > 1)
		(params->UsingYIndex)++;

	if (params->UsingYIndex)
		{
			params->Y1=p1x;
			params->X1=p1y;
			params->Y2=p2x;
			params->X2=p2y;
		}
	else
		{
			params->X1=p1x;
			params->Y1=p1y;
			params->X2=p2x;
			params->Y2=p2y;
		}

	 if ((p2x - p1x) * (p2y - p1y) < 0)
		{
			params->Flipped = 1;
			params->Y1 = -params->Y1;
			params->Y2 = -params->Y2;
		}
	else
		params->Flipped = 0;

	if (params->X2 > params->X1)
		params->Increment = 1;
	else
		params->Increment = -1;

	params->DeltaX=params->X2-params->X1;
	params->DeltaY=params->Y2-params->Y1;

	params->IncrE=2*params->DeltaY*params->Increment;
	params->IncrNE=2*(params->DeltaY-params->DeltaX)*params->Increment;
	params->DTerm=(2*params->DeltaY-params->DeltaX)*params->Increment;

	params->XIndex = params->X1;
	params->YIndex = params->Y1;
}

void get_current_point(bresenham_param_t *params, int *x, int *y) {
	if (params->UsingYIndex) {
        *y = params->XIndex;
        *x = params->YIndex;
        if (params->Flipped)
            *x = -*x;
    }
	else {
        *x = params->XIndex;
        *y = params->YIndex;
        if (params->Flipped)
            *y = -*y;
    }
}

int get_next_point(bresenham_param_t *params) {
	if (params->XIndex == params->X2) {
        return 0;
    }
	params->XIndex += params->Increment;
	if (params->DTerm < 0 || (params->Increment < 0 && params->DTerm <= 0))
		params->DTerm += params->IncrE;
	else {
        params->DTerm += params->IncrNE;
        params->YIndex += params->Increment;
	}
	return 1;
}



int IsValidLineSegment(double x0, double y0, double x1, double y1, double*	map,
			 int x_size, int y_size) {
	bresenham_param_t params;
	int nX, nY; 
	short unsigned int nX0, nY0, nX1, nY1;

	//printf("checking link <%f %f> to <%f %f>\n", x0,y0,x1,y1);
		
	//make sure the line segment is inside the environment
	if(x0 < 0 || x0 >= x_size ||
		x1 < 0 || x1 >= x_size ||
		y0 < 0 || y0 >= y_size ||
		y1 < 0 || y1 >= y_size)
		return 0;

	ContXY2Cell(x0, y0, &nX0, &nY0, x_size, y_size);
	ContXY2Cell(x1, y1, &nX1, &nY1, x_size, y_size);

	//printf("checking link <%d %d> to <%d %d>\n", nX0,nY0,nX1,nY1);

	//iterate through the points on the segment
	get_bresenham_parameters(nX0, nY0, nX1, nY1, &params);
	do {
		get_current_point(&params, &nX, &nY);
		if(map[GETMAPINDEX(nX,nY,x_size,y_size)] == 1)
			return 0;
	} while (get_next_point(&params));

	return 1;
}

int IsValidArmConfiguration(double* angles, int numofDOFs, double*	map,
			 int x_size, int y_size) {
    double x0,y0,x1,y1;
    int i;
		
	 //iterate through all the links starting with the base
	x1 = ((double)x_size)/2.0;
	y1 = 0;
	for(i = 0; i < numofDOFs; i++){
		//compute the corresponding line segment
		x0 = x1;
		y0 = y1;
		x1 = x0 + LINKLENGTH_CELLS*cos(2*PI-angles[i]);
		y1 = y0 - LINKLENGTH_CELLS*sin(2*PI-angles[i]);

		//check the validity of the corresponding line segment
		if(!IsValidLineSegment(x0,y0,x1,y1,map,x_size,y_size))
			return 0;
	}    
	return 1;
}

//*******************************************Common helper fcns Start*****************************************************//
double gen_rand_num(double lower_bound, double upper_bound) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(lower_bound, upper_bound);
	return dis(gen);
}

// double* gen_rand_config(int numofDOFs, double* map, int x_size, int y_size) {
// 	double* q = new double[numofDOFs];

//     while (!IsValidArmConfiguration(q, numofDOFs, map, x_size, y_size)) {
//         for (int i = 0; i < numofDOFs; i++) {
//             q[i] = gen_rand_num(0, 2*PI);
//         }
//     }

// 	return q;
// }

double* gen_rand_config(int numofDOFs, double* map, int x_size, int y_size) {
    double* q = new double[numofDOFs];

    for (int i = 0; i < numofDOFs; i++) {
        q[i] = gen_rand_num(0, 2*PI);
    }

	return q;
}
//*******************************************Common helper fcns End*******************************************************//

//**********************************************RRT Helper fcns Start****************************************************//
struct TreeNode {
	double* q;
    double cost;
	TreeNode* parent;
	TreeNode* leftchild;
	TreeNode* rightchild;
	TreeNode(double* q, double cost = 0.0, TreeNode* parent= nullptr, TreeNode* leftchild= nullptr, TreeNode* rightchild= nullptr) : q(q), cost(cost) ,parent(parent),  leftchild(leftchild), rightchild(rightchild){}
};

TreeNode* insert_KDtree_node(TreeNode* root, double* q, int depth, int numJoints, double cost) {
    // Base case: If root is nullptr, create and return a new node
    if (root == nullptr) {
        root = new TreeNode(q, cost);
        cout << "Root node inserted" << endl;
        return root; // Return the root node which is the newly created node
    }

    TreeNode* current = root;  // Pointer to traverse the tree
    TreeNode* parent = nullptr; // Pointer to store the parent of the current node
    int currentDepth = depth;   // Maintain current depth to calculate the cutting dimension

    while (current != nullptr) {
        int cutDim = currentDepth % numJoints;  // Determine the cutting dimension
        parent = current;  // Keep track of the parent node

        // Compare q[cutDim] to decide whether to go left or right
        if (q[cutDim] < current->q[cutDim]) {
            // Move to the left child
            current = current->leftchild;
        } else {
            // Move to the right child
            current = current->rightchild;
        }

        // Increase the depth at each level
        currentDepth++;
    }

    // Now `parent` points to the node where we need to insert the new node
    int cutDim = (currentDepth - 1) % numJoints;  // Get the correct dimension at this depth

    // Insert the new node as either the left or right child
    if (q[cutDim] < parent->q[cutDim]) {
        parent->leftchild = new TreeNode(q, cost, parent);  // Insert as the left child
        // cout << "Left child node inserted" << endl;
        return parent->leftchild;  // Return the inserted node
    } else {
        parent->rightchild = new TreeNode(q, cost, parent);  // Insert as the right child
        // cout << "Right child node inserted" << endl;
        return parent->rightchild;  // Return the inserted node
    }
}

double configDistance(double* q1, double* q2, int numJoints) {
	double dist = 0;
	for (int i = 0; i < numJoints; i++) {
        // double diff = fabs(q1[i] - q2[i]);
		dist += pow(q1[i] - q2[i], 2);
        // dist += fmin(diff, fabs(2*PI - diff));
	}
	return sqrt(dist);
    // return dist;
}

// Recursive function to find nearest neighbor in a KD-tree
TreeNode* nearestNeighbor(TreeNode* root, double* q, int numJoints, int level, TreeNode* bestNode = nullptr, double bestDist = INFINITY) {
    if (root == nullptr) {
        return bestNode;
    }

    // Calculate the distance to the current node
    double currentDist = configDistance(q, root->q, numJoints);

    // Update best node and best distance if current node is closer
    if (currentDist < bestDist) {
        bestDist = currentDist;
        bestNode = root;
    }

    // Determine the current dimension to compare (cycling through dimensions)
    int cutDim = level % numJoints;

    // Recursively search the closer subtree
    TreeNode* nextBranch = nullptr;
    TreeNode* oppositeBranch = nullptr;

    if (q[cutDim] < root->q[cutDim]) {
        nextBranch = root->leftchild;
        oppositeBranch = root->rightchild;
    } else {
        nextBranch = root->rightchild;
        oppositeBranch = root->leftchild;
    }

    // Recurse on the side of the tree that is closer to the point
    bestNode = nearestNeighbor(nextBranch, q, numJoints, level + 1, bestNode, bestDist);

    // If the current best distance crosses the splitting plane, explore the other side
    if (fabs(q[cutDim] - root->q[cutDim]) < bestDist) {
        bestNode = nearestNeighbor(oppositeBranch, q, numJoints, level + 1, bestNode, bestDist);
    }

    return bestNode;
}

TreeNode* extend_KDtree(TreeNode* root, double* q, int numJoints, double* map, int x_size, int y_size, double epsilon, double step) {
    // Find the nearest neighbor to the random configuration
    TreeNode* nearest = nearestNeighbor(root, q, numJoints, 0);
    if (!nearest) {
        cout << "Nearest neighbor not found" << endl;
        return nullptr;
    }

    // Extend the tree toward the random configuration
    double* q_new = new double[numJoints];
    double* q_prev = new double[numJoints];
    double cost_new = nearest->cost + configDistance(nearest->q, q, numJoints);
    copy(nearest->q, nearest->q + numJoints, q_prev);

    for (double stepsize = step; stepsize <= epsilon; stepsize += step) {
        for (int i = 0; i < numJoints; i++) {
            q_new[i] = nearest->q[i] + stepsize * (q[i] - nearest->q[i])/configDistance(nearest->q, q, numJoints);
        }

        if (!IsValidArmConfiguration(q_new, numJoints, map, x_size, y_size)) {
            copy(q_prev, q_prev + numJoints, q_new);
            break;
            // return nullptr;
        } else {
            copy(q_new, q_new + numJoints, q_prev);
        }

        if (configDistance(nearest->q, q, numJoints) < stepsize) {
            copy(q, q + numJoints, q_new);
            break;
        }
    }
    
    // Insert the new configuration into the tree
    cost_new = nearest->cost + configDistance(nearest->q, q_new, numJoints);
    TreeNode* new_node = insert_KDtree_node(root, q_new, 0, numJoints, cost_new);  

    return new_node; // Return the newly created node
}

// Function to backtrack from the goal node to the start node and return the path
vector<TreeNode*> backtrackPath(TreeNode* goal_node) {
    vector<TreeNode*> path;
    
    // Start at the goal node and backtrack to the root using the parent pointers
    TreeNode* current = goal_node;
    while (current != nullptr) {
        path.push_back(current);  // Add the current node to the path
        current = current->parent;  // Move to the parent node
    }

    // The path is currently from goal to start, so we reverse it to go from start to goal
    std::reverse(path.begin(), path.end());

    return path;  // Return the path from start to goal
}

//*****************************************RRT Helper fcns End*******************************************************//

//*****************************************RRT Star Helper fcns Start***********************************************//

bool IsValidInterplolation(double* q1, double* q2, int numJoints, double* map, int x_size, int y_size, double step) {
    double dist = configDistance(q1, q2, numJoints);
    double* q = new double[numJoints];
    for (double t = 0; t < dist; t += step) {
        for (int i = 0; i < numJoints; i++) {
            q[i] = q1[i] + t/dist * (q2[i] - q1[i]);
        }
        if (!IsValidArmConfiguration(q, numJoints, map, x_size, y_size)) {
            return false;
        }
    }
    return true;
}

// Recursive function to count nodes in a KD-tree
int countNodes(TreeNode* node) {
    if (node == nullptr) {
        return 0;
    }
    // Count current node + count of left subtree + count of right subtree
    return 1 + countNodes(node->leftchild) + countNodes(node->rightchild);
}

vector<TreeNode*> findNeighbours(TreeNode* root, double* q_new, double radius, int numJoints) {
    vector<TreeNode*> neighbors;
    vector<TreeNode*> queue;
    queue.push_back(root);

    while (!queue.empty()) {
        TreeNode* current = queue.back();
        queue.pop_back();

        if (configDistance(current->q, q_new, numJoints) < radius) {
            neighbors.push_back(current);
        }

        if (current->leftchild != nullptr) {
            queue.push_back(current->leftchild);
        }
        if (current->rightchild != nullptr) {
            queue.push_back(current->rightchild);
        }
    }

    return neighbors;
}

TreeNode* extend_KDtree_star(TreeNode* root, double* q , int numJoints, double* map, int x_size, int y_size, double epsilon, double step, double eta) {

    // Find the nearest neighbor to the random configuration
    TreeNode* nearest = nearestNeighbor(root, q, numJoints, 0);
    if (!nearest) {
        cout << "Nearest neighbor not found" << endl;
        return nullptr;
    }

    // Extend the tree toward the random configuration
    double* q_new = new double[numJoints];
    double* q_prev = new double[numJoints];
    double cost_new = nearest->cost;
    copy(nearest->q, nearest->q + numJoints, q_prev);

    double stepsize = step;
    for (double stepsize = step; stepsize <= epsilon; stepsize += step) {
        for (int i = 0; i < numJoints; i++) {
            q_new[i] = nearest->q[i] + stepsize * (q[i] - nearest->q[i])/configDistance(nearest->q, q, numJoints);
        }

        if (!IsValidArmConfiguration(q_new, numJoints, map, x_size, y_size)) {
            copy(q_prev, q_prev + numJoints, q_new);
            break;
        } else {
            copy(q_new, q_new + numJoints, q_prev);
        }

        if (configDistance(nearest->q, q, numJoints) < stepsize) {
            copy(q, q + numJoints, q_new);
            break;
        }
    }
    
    // Insert the new configuration into the tree
    cost_new = nearest->cost + configDistance(nearest->q, q_new, numJoints);
    
    TreeNode* best_parent = nearest;
    int numNodes = countNodes(root);
    double radius = eta * sqrt(log(numNodes+1)/(numNodes+1));

    vector<TreeNode*> neighbors = findNeighbours(root, q_new, radius, numJoints);

    for (TreeNode* neighbour : neighbors) {
        double cost = neighbour->cost + configDistance(neighbour->q, q_new, numJoints);
        if (cost < cost_new && IsValidInterplolation(neighbour->q, q_new, numJoints, map, x_size, y_size, step)) {
            best_parent = neighbour;
            cost_new = cost;
        }
    }

    TreeNode* new_node = insert_KDtree_node(root, q_new, 0, numJoints, cost_new);
    return new_node;
}

//*****************************************RRT Star Helper fcns end***********************************************//

//******************************************PRM Helper fcns Start****************************************************//
// Structure for storing graph nodes
struct GraphNode {
    double* q;   // Configuration (joint angles)
    double cost;
    vector<GraphNode*> neighbors; // Neighboring nodes in the graph
    GraphNode(double* q, double cost = INFINITY) : q(q), cost(cost) {}
};

// Function to connect nearby nodes in the roadmap
void connectGraphNodes(GraphNode* node, vector<GraphNode*>& G, int numJoints, double* map, int x_size, int y_size, double connection_radius, double step) {
	// Iterate through all nodes in the roadmap
	for (GraphNode* neighbor : G) {
		// Skip the same node
		// Check if the distance between the nodes is less than the connection radius
		if (node != neighbor && configDistance(node->q, neighbor->q, numJoints) < connection_radius) {
            if (IsValidInterplolation(node->q, neighbor->q, numJoints, map, x_size, y_size, step)) {
                node->neighbors.push_back(neighbor);
                neighbor->neighbors.push_back(node);
            }
		}
	}
}

vector<GraphNode*> graphSearch(GraphNode* startNode, GraphNode* goalNode, int numofDOFs, const vector<GraphNode*>& roadmap) {
    std::unordered_map<GraphNode*, double> dist;
    std::unordered_map<GraphNode*, GraphNode*> prev;
    std::priority_queue<pair<double, GraphNode*>, vector<pair<double, GraphNode*>>, std::greater<>> pq;

    for (auto& node : roadmap) {
        dist[node] = std::numeric_limits<double>::infinity();
		prev[node] = nullptr;  // Initialize previous nodes to null
    }
    dist[startNode] = 0;
    pq.push({0, startNode});

    while (!pq.empty()) {
        GraphNode* current = pq.top().second;
        pq.pop();

		// Debug: Output current node
        cout << "Exploring node with config: ";
        for (int i = 0; i < numofDOFs; i++) {
            cout << current->q[i] << " ";
        }
        cout << endl;

        if (current == goalNode) {
            // Goal reached, backtrack to construct the path
            vector<GraphNode*> path;
            for (GraphNode* node = goalNode; node != nullptr; node = prev[node]) {
				if (node == nullptr) {
					cout << "Error: node is null during path construction." << endl;
					break;
				}
                path.push_back(node);
            }
            std::reverse(path.begin(), path.end());
            return path;
        }

        // Explore neighbors
        for (GraphNode* neighbor : current->neighbors) {
            double alt = dist[current] + configDistance(current->q, neighbor->q, numofDOFs);
            if (alt < dist[neighbor]) {
                dist[neighbor] = alt;
                prev[neighbor] = current;
                pq.push({alt, neighbor});

				// Debug: Output updated neighbor
                cout << "Updating neighbor: ";
                for (int i = 0; i < numofDOFs; i++) {
                    cout << neighbor->q[i] << " ";
                }
                cout << " with dist: " << alt << endl;
            }
        }
    }
    return {}; // Return empty path if no path found
}

//******************************************PRM Helper fcns End******************************************************//


//*******************************************************************************************************************//
//                                                                                                                   //
//                                          DEFAULT PLANNER FUNCTION                                                 //
//                                                                                                                   //
//*******************************************************************************************************************//

void planner(
    double* map,
	int x_size,
	int y_size,
	double* armstart_anglesV_rad,
	double* armgoal_anglesV_rad,
    int numofDOFs,
    double*** plan,
    int* planlength)
{
	//no plan by default
	*plan = NULL;
	*planlength = 0;
		
    //for now just do straight interpolation between start and goal checking for the validity of samples

    double distance = 0;
    int i,j;
    for (j = 0; j < numofDOFs; j++){
        if(distance < fabs(armstart_anglesV_rad[j] - armgoal_anglesV_rad[j]))
            distance = fabs(armstart_anglesV_rad[j] - armgoal_anglesV_rad[j]);
    }
    int numofsamples = (int)(distance/(PI/20));
    if(numofsamples < 2){
        printf("the arm is already at the goal\n");
        return;
    }
    *plan = (double**) malloc(numofsamples*sizeof(double*));
    int firstinvalidconf = 1;
    for (i = 0; i < numofsamples; i++){
		cout << "Sample " << i << endl;
        (*plan)[i] = (double*) malloc(numofDOFs*sizeof(double)); 
        for(j = 0; j < numofDOFs; j++){
            (*plan)[i][j] = armstart_anglesV_rad[j] + ((double)(i)/(numofsamples-1))*(armgoal_anglesV_rad[j] - armstart_anglesV_rad[j]);
        }
        if(!IsValidArmConfiguration((*plan)[i], numofDOFs, map, x_size, y_size) && firstinvalidconf) {
            firstinvalidconf = 1;
            printf("ERROR: Invalid arm configuration!!!\n");
        }
    }    
    *planlength = numofsamples;
    
    return;
}

//*******************************************************************************************************************//
//                                                                                                                   //
//                                              RRT IMPLEMENTATION                                                   //
//                                                                                                                   //
//*******************************************************************************************************************//

static void plannerRRT(
    double *map,
    int x_size,
    int y_size,
    double *armstart_anglesV_rad,
    double *armgoal_anglesV_rad,
    int numofDOFs,
    double ***plan,
    int *planlength)
{

	int numofsamples = (int)1e5;
	double goal_bias = 0.6;  // Probability of sampling the goal
    double epsilon = 3;  // Reduced step size for finer movements
	double goal_threshold = 1e-3;
    double step = 0.1;
	cout << "Number of samples: " << numofsamples << endl;

	cout << "Initializing RRT..." << endl;

	TreeNode* root = insert_KDtree_node(nullptr, armstart_anglesV_rad, 0, numofDOFs, 0);
    TreeNode* goal_node = nullptr;

	for (int i=0; i <= numofsamples; i++){

		cout << "Sample " << i << endl;
		double* q_rand;

		if (((double)rand() / RAND_MAX) < goal_bias) {
            q_rand = armgoal_anglesV_rad;  // Use the goal configuration
        } else {
            q_rand = gen_rand_config(numofDOFs, map, x_size, y_size);
            if (q_rand) {
                // Valid configuration generated
            } else {
                cout << "Invalid configuration generated \nAttempting to generate again... " << endl;
                continue;  
            }
        }

		// Try to extend the tree toward the random sample
        TreeNode* newNode = extend_KDtree(root, q_rand, numofDOFs, map, x_size, y_size, epsilon, step);

        if (!newNode){
            cout << "Failed to extend tree" << endl;
            continue;
        }

        if (configDistance(newNode->q, armgoal_anglesV_rad, numofDOFs) < goal_threshold) {
            // Set newNode as the parent of the goal_node
            // goal_node = new TreeNode(armgoal_anglesV_rad, 0.0, newNode);
            goal_node = newNode;
            cout << "Goal reached in tree" << endl;
            break;
        }
	}

	if (goal_node) {
        cout << "Extracting path..." << endl;
        vector<TreeNode*> path = backtrackPath(goal_node);
        *planlength = path.size();

        if (*planlength > 0) {
            *plan = (double**)malloc(*planlength * sizeof(double*));
            if (!*plan) {
                cout << "Failed to allocate memory for plan" << endl;
                return;
            }

            for (int i = 0; i < *planlength; i++) {
                (*plan)[i] = new double[numofDOFs]; 
                if (!(*plan)[i]) {
                    cout << "Failed to allocate memory for plan[" << i << "]" << endl;
                    // Clean up previously allocated memory
                    for (int j = 0; j < i; j++) {
                        delete[] (*plan)[j];
                    }
                    free(*plan);
                    *plan = NULL;
                    *planlength = 0;
                    return;
                }
                for (int j = 0; j < numofDOFs; j++) {
                    (*plan)[i][j] = path[i]->q[j];
                }
            }
            cout << "Path found via RRT" << endl;
            cout << "Path length: " << *planlength << endl;
        } else {
            cout << "No valid path found via RRT" << endl;
        }
    } else {
        cout << "Failed to find a path to the goal via RRT." << endl;
        cout << "Attempting default planner..." << endl;
        planner(map, x_size, y_size, armstart_anglesV_rad, armgoal_anglesV_rad, numofDOFs, plan, planlength);
    }

    cout << "RRT planning completed" << endl;
}

//*******************************************************************************************************************//
//                                                                                                                   //
//                                         RRT CONNECT IMPLEMENTATION                                                //
//                                                                                                                   //
//*******************************************************************************************************************//

static void plannerRRTConnect(
    double *map,
    int x_size,
    int y_size,
    double *armstart_anglesV_rad,
    double *armgoal_anglesV_rad,
    int numofDOFs,
    double ***plan,
    int *planlength)
{
    /* TODO: Replace with your implementation */

	int numofsamples = (int)1e5;
	double sampling_bias = 0.2;  // Probability of sampling the goal
    double epsilon = 0.6;  // Reduced step size for finer movements
	double target_threshold = 1e-3;
    double step = 0.1;
    bool path_found = false;
    bool switchFlag = true;

	cout << "Initializing RRT Connect..." << endl;

    TreeNode* start_root = insert_KDtree_node(nullptr, armstart_anglesV_rad, 0, numofDOFs, 0);
    TreeNode* goal_root = insert_KDtree_node(nullptr, armgoal_anglesV_rad, 0, numofDOFs, 0);
    TreeNode* newNode_s = nullptr;
    TreeNode* newNode_g = nullptr;

    for (int i=0; i <= numofsamples; i++){
		cout << "Sample " << i << endl;
		double* q_rand;

		if (((double)rand() / RAND_MAX) < sampling_bias) {
            if (switchFlag) {q_rand = armgoal_anglesV_rad;}  // Use the goal configuration
            else {q_rand = armstart_anglesV_rad;}
        } else {
            q_rand = gen_rand_config(numofDOFs, map, x_size, y_size);
			if (!q_rand) {
                cout << "Failed to generate random configuration" << endl;
                continue;
            }
        }

		// Try to extend the tree toward the random sample
        if (switchFlag) {
            newNode_s = extend_KDtree(start_root, q_rand, numofDOFs, map, x_size, y_size, epsilon, step);
            newNode_g = extend_KDtree(goal_root, newNode_s->q, numofDOFs, map, x_size, y_size, INFINITY, step);
            if (newNode_g == nullptr || newNode_s == nullptr) {
                cout << "Failed to extend tree" << endl;
                switchFlag = false;
                continue;
            }

            if (configDistance(newNode_s->q, newNode_g->q, numofDOFs) < target_threshold) {
                // Set newNode as the parent of the goal_node
                path_found = true;
                cout << "Goal reached in tree" << endl;
                break;
            }
            switchFlag = false;
        } else {
            newNode_g = extend_KDtree(goal_root, q_rand, numofDOFs, map, x_size, y_size, epsilon, step);
            newNode_s = extend_KDtree(start_root, newNode_g->q, numofDOFs, map, x_size, y_size, INFINITY, step);

            if (newNode_g == nullptr || newNode_s == nullptr) {
                cout << "Failed to extend tree" << endl;
                switchFlag = true;
                continue;
            }

            if (configDistance(newNode_s->q, newNode_g->q, numofDOFs) < target_threshold) {
                // Set newNode as the parent of the goal_node
                path_found = true;
                cout << "Goal reached in tree" << endl;
                break;
            }
            switchFlag = true;
        }
	}

    if (path_found) {
        cout << "Extracting path..." << endl;

        // Backtrack from the goal node in the goal tree
        vector<TreeNode*> path_goal = backtrackPath(newNode_g);  // From goal to connection point

        // Backtrack from the start node in the start tree
        vector<TreeNode*> path_start = backtrackPath(newNode_s);  // From start to connection point

        // Combine the paths
        vector<TreeNode*> full_path = path_start;
        full_path.insert(full_path.end(), path_goal.rbegin(), path_goal.rend());  // Concatenate in reverse order

        // Convert the path to the plan array format
        *planlength = full_path.size();

        *plan = (double**)malloc(*planlength * sizeof(double*));
        for (int i = 0; i < *planlength; i++) {
            (*plan)[i] = new double[numofDOFs];
            for (int j = 0; j < numofDOFs; j++) {
                (*plan)[i][j] = full_path[i]->q[j];
            }
        }

        cout << "Path found and extracted via RRT Connect" << endl;
        cout << "Path length: " << *planlength << endl;
    } else {
        cout << "Failed to find a path to the goal via RRT Connect" << endl;
        // planner(map, x_size, y_size, armstart_anglesV_rad, armgoal_anglesV_rad, numofDOFs, plan, planlength);
    }
}

//*******************************************************************************************************************//
//                                                                                                                   //
//                                           RRT STAR IMPLEMENTATION                                                 //
//                                                                                                                   //
//*******************************************************************************************************************//

static void plannerRRTStar(
    double *map,
    int x_size,
    int y_size,
    double *armstart_anglesV_rad,
    double *armgoal_anglesV_rad,
    int numofDOFs,
    double ***plan,
    int *planlength)
{
    /* TODO: Replace with your implementation */
    // planner(map, x_size, y_size, armstart_anglesV_rad, armgoal_anglesV_rad, numofDOFs, plan, planlength);

    int numofsamples = (int)1e5;
	double goal_bias = 0.4;  // Probability of sampling the goal
    double epsilon = 3;  // Reduced step size for finer movements
	double goal_threshold = 1e-3;
    double step = 0.1;
    double eta = 2.5;

    TreeNode* root = insert_KDtree_node(nullptr, armstart_anglesV_rad, 0, numofDOFs, 0.0);
    TreeNode* goal_node = nullptr;

    for (int i=0; i <= numofsamples; i++){
        cout << "Sample " << i << endl;
		double* q_rand;

		if (((double)rand() / RAND_MAX) < goal_bias) {
            q_rand = armgoal_anglesV_rad;  // Use the goal configuration
        } else {
            q_rand = gen_rand_config(numofDOFs, map, x_size, y_size);
			if (!q_rand) {
                cout << "Failed to generate random configuration" << endl;
                continue;
            }
        }

        // extend and rewire the tree
        TreeNode* newNode = extend_KDtree_star(root, q_rand, numofDOFs, map, x_size, y_size, epsilon, step, eta);
        if (!newNode){
            cout << "Failed to extend tree" << endl;
            continue;
        }

        if (configDistance(newNode->q, armgoal_anglesV_rad, numofDOFs) < goal_threshold) {
            // Set newNode as the parent of the goal_node
            goal_node = new TreeNode(armgoal_anglesV_rad, newNode->cost + configDistance(newNode->q, armgoal_anglesV_rad, numofDOFs), newNode);
            cout << "Goal reached in tree" << endl;
            break;
        }
    }

    if (goal_node) {
        cout << "Extracting path..." << endl;
        vector<TreeNode*> path = backtrackPath(goal_node);
        *planlength = path.size();

        if (*planlength > 0) {
            *plan = (double**)malloc(*planlength * sizeof(double*));
            if (!*plan) {
                cout << "Failed to allocate memory for plan" << endl;
                return;
            }

            for (int i = 0; i < *planlength; i++) {
                (*plan)[i] = new double[numofDOFs]; 
                if (!(*plan)[i]) {
                    cout << "Failed to allocate memory for plan[" << i << "]" << endl;
                    // Clean up previously allocated memory
                    for (int j = 0; j < i; j++) {
                        delete[] (*plan)[j];
                    }
                    free(*plan);
                    *plan = NULL;
                    *planlength = 0;
                    return;
                }
                for (int j = 0; j < numofDOFs; j++) {
                    (*plan)[i][j] = path[i]->q[j];
                }
            }
            cout << "Path found via RRT Star" << endl;
            cout << "Path length: " << *planlength << endl;
        } else {
            cout << "No valid path found via RRT Star" << endl;
        }
    } else {
        cout << "Failed to find a path to the goal via RRT Star." << endl;
        cout << "Attempting to plan using default planner..." << endl;
        // planner(map, x_size, y_size, armstart_anglesV_rad, armgoal_anglesV_rad, numofDOFs, plan, planlength);
    }

}

//*******************************************************************************************************************//
//                                                                                                                   //
//                                              PRM IMPLEMENTATION                                                   //
//                                                                                                                   //
//*******************************************************************************************************************//

static void plannerPRM(
    double *map,
    int x_size,
    int y_size,
    double *armstart_anglesV_rad,
    double *armgoal_anglesV_rad,
    int numofDOFs,
    double ***plan,
    int *planlength)
{
    /* TODO: Replace with your implementation */

    int numofsamples = (int)1e5;
	double connection_radius = 1.0; // Maximum connection distance
    double goal_bias = 0.1;  // Probability of sampling the goal
    double step = 0.1;

	// Step 1: Build Roadmap (Sample valid configurations and connect them)
    vector<GraphNode*> G;

	// Add the start and goal as nodes in the roadmap
    GraphNode* startNode = new GraphNode(armstart_anglesV_rad, 0);
    GraphNode* goalNode = new GraphNode(armgoal_anglesV_rad);
    G.push_back(startNode);
    G.push_back(goalNode);

	// Add random samples to the roadmap which are collision-free
	for (int i = 0; i < numofsamples; i++) {
        double* q_rand;

		if (((double)rand() / RAND_MAX) < goal_bias) {
            q_rand = armgoal_anglesV_rad;  // Use the goal configuration
        } else {
            q_rand = gen_rand_config(numofDOFs, map, x_size, y_size);
			if (q_rand && IsValidArmConfiguration(q_rand, numofDOFs, map, x_size, y_size)) {
                G.push_back(new GraphNode(q_rand));
            } else {
                cout << "Invalid configuration generated \nAttempting to generate again... " << endl;
                continue;   
            }
        }
    }

	cout << "Samples generated: "<< G.size() << endl;

	// Step 2: Connect nearby nodes in the roadmap
	for (GraphNode*& node : G) {
    	connectGraphNodes(node, G, numofDOFs, map, x_size, y_size, connection_radius, step);
	}
	cout << "Roadmap generated with nodes: " << G.size() << endl;

	// Step 3: Perform graph search to find a path from start to goal
    vector<GraphNode*> path = graphSearch(startNode, goalNode, numofDOFs, G);

    // Step 4: Store the result
    if (!path.empty()) {
		cout << "Path found via PRM" << endl;
		cout << "Number of nodes:" << G.size() << endl;
		cout << "Path length: " << path.size() << endl;
        *planlength = path.size();
        *plan = (double**)malloc(*planlength * sizeof(double*));
        for (int i = 0; i < *planlength; i++) {
            (*plan)[i] = path[i]->q;
        }
    } else {
        cout << "Failed to find a path to the goal via PRM." << endl;
        cout << "Attempting to plan using default planner..." << endl;
        // planner(map, x_size, y_size, armstart_anglesV_rad, armgoal_anglesV_rad, numofDOFs, plan, planlength);
    }
}

// Function to save all start and goal pairs to a single file
void saveAllStartGoalPairsToFile(const std::vector<double*>& startPositions, const std::vector<double*>& goalPositions, int numOfDOFs, const std::string& filename) {
    std::ofstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    for (int i = 0; i < startPositions.size(); i++) {
        file << "Start Configuration " << (i + 1) << ":\n";
        for (int j = 0; j < numOfDOFs; j++) {
            file << startPositions[i][j];
            if (j < numOfDOFs - 1) file << ",";
        }
        file << "\n";

        file << "Goal Configuration " << (i + 1) << ":\n";
        for (int j = 0; j < numOfDOFs; j++) {
            file << goalPositions[i][j];
            if (j < numOfDOFs - 1) file << ",";
        }
        file << "\n\n";  // Separate pairs by an extra newline
    }

    file.close();
    std::cout << "All start-goal pairs saved to " << filename << std::endl;
}
//*******************************************************************************************************************//
//                                                                                                                   //
//                                                MAIN FUNCTION                                                      //
//                                                                                                                   //
//*******************************************************************************************************************//

/** Your final solution will be graded by an grading script which will
 * send the default 6 arguments:
 *    map, numOfDOFs, commaSeparatedStartPos, commaSeparatedGoalPos, 
 *    whichPlanner, outputFilePath
 * An example run after compiling and getting the planner.out executable
 * >> ./planner map1.txt 5 1.57,0.78,1.57,0.78,1.57 0.392,2.35,3.14,2.82,4.71 0 output.txt
 * See the hw handout for full information.
 * If you modify this for testing (e.g. to try out different hyper-parameters),
 * make sure it can run with the original 6 commands.
 * Programs that do not will automatically get a 0.
 * */
int main(int argc, char** argv) {
	double* map;
	int x_size, y_size;

    std::string mapDirPath = MAPS_DIR;
    std::string mapFilePath = mapDirPath + "/" + argv[1];
    std::cout << "Reading problem definition from: " << mapFilePath << std::endl;
	tie(map, x_size, y_size) = loadMap(mapFilePath);
	const int numOfDOFs = std::stoi(argv[2]);
	double* startPos = doubleArrayFromString(argv[3]);
	double* goalPos = doubleArrayFromString(argv[4]);
	int whichPlanner = std::stoi(argv[5]);

    std::string outputDir = OUTPUT_DIR;
	string outputFile = outputDir + "/" + argv[6];
	std::cout << "Writing solution to: " << outputFile << std::endl;

	if(!IsValidArmConfiguration(startPos, numOfDOFs, map, x_size, y_size)||
			!IsValidArmConfiguration(goalPos, numOfDOFs, map, x_size, y_size)) {
		throw runtime_error("Invalid start or goal configuration!\n");
	}

	///////////////////////////////////////
	//// Feel free to modify anything below. Be careful modifying anything above.


    // Generate 20 random start and goal positions
    // vector<double*> startPositions;
    // vector<double*> goalPositions;
    // int i =0 ;
    // while(i<20){
    //     double* start = gen_rand_config(numOfDOFs, map, x_size, y_size);
    //     double* goal = gen_rand_config(numOfDOFs, map, x_size, y_size);
    //     if(IsValidArmConfiguration(start, numOfDOFs, map, x_size, y_size) && IsValidArmConfiguration(goal, numOfDOFs, map, x_size, y_size)){
    //         startPositions.push_back(start);
    //         goalPositions.push_back(goal);
    //         i++;
    //     }
    // }

    // string startGoalFile = string(OUTPUT_DIR) + "/start_goal_pairs.txt";
    // saveAllStartGoalPairsToFile(startPositions, goalPositions, numOfDOFs, startGoalFile);

    // Start timing for a single run
    auto start = std::chrono::high_resolution_clock::now();

	double** plan = NULL;
	int planlength = 0;

    // Call the corresponding planner function
    if (whichPlanner == PRM)
    {
        plannerPRM(map, x_size, y_size, startPos, goalPos, numOfDOFs, &plan, &planlength);
    }
    else if (whichPlanner == RRT)
    {
        plannerRRT(map, x_size, y_size, startPos, goalPos, numOfDOFs, &plan, &planlength);
    }
    else if (whichPlanner == RRTCONNECT)
    {
        plannerRRTConnect(map, x_size, y_size, startPos, goalPos, numOfDOFs, &plan, &planlength);
    }
    else if (whichPlanner == RRTSTAR)
    {
        plannerRRTStar(map, x_size, y_size, startPos, goalPos, numOfDOFs, &plan, &planlength);
    }
    else
    {
        planner(map, x_size, y_size, startPos, goalPos, numOfDOFs, &plan, &planlength);
    }

    // End timing and calculate the duration
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution time for the run: " << duration.count() << " seconds" << std::endl;


	//// Feel free to modify anything above.
	//// If you modify something below, please change it back afterwards as my 
	//// grading script will not work and you will recieve a 0.
	///////////////////////////////////////

    // Your solution's path should start with startPos and end with goalPos
    if (!equalDoubleArrays(plan[0], startPos, numOfDOFs) || 
    	!equalDoubleArrays(plan[planlength-1], goalPos, numOfDOFs)) {
		throw std::runtime_error("Start or goal position not matching");
	}

	/** Saves the solution to output file
	 * Do not modify the output log file output format as it is required for visualization
	 * and for grading.
	 */
	std::ofstream m_log_fstream;
	m_log_fstream.open(outputFile, std::ios::trunc); // Creates new or replaces existing file
	if (!m_log_fstream.is_open()) {
		throw std::runtime_error("Cannot open file");
	}
	m_log_fstream << mapFilePath << endl; // Write out map name first
	/// Then write out all the joint angles in the plan sequentially
	for (int i = 0; i < planlength; ++i) {
		for (int k = 0; k < numOfDOFs; ++k) {
			m_log_fstream << plan[i][k] << ",";
		}
		m_log_fstream << endl;
	}
}
