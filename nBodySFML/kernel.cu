
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "thrust\device_vector.h"
#include <curand.h>
#include <curand_kernel.h>

#include <stdio.h>

#include "SFML\Graphics.hpp"
#include "Node.h"
#include <ctime>
#include <math.h>
#include <cmath>
#include <vector>
#include <omp.h>
#include <thread>
#include <random>
#include <chrono>


#define _PI 3.14159265      //Pi, used for calculations and rounded to 8 decimal places. 
#define _GRAV_CONST 0.1     //the gravitational constant. This is the timestep between each frame. Lower for slower but more accurate simulations

void BodyAttraction(std::vector<Body*> &pBodies, float pSoftener);                                                                       //Attracts each body to each other body in the given vector of pointers to body objects
void CalculateForceNode(Body* bi, Node* bj, float pSoftener);                                                                           //Calculate force exerted on body from node
void CalculateForce(Body* bi, Body* bj, float pSoftener);                                                                               //Calculate force exerted on eachother between two bodies
//Body* CreateBody(float px, float py, float pmass, float pvx = 0, float pvy = 0);                                                        //return a pointer to new body object defined on the heap with given paramiters
void DeleteBodies(std::vector<Body*> &pBodies);                                                                                          //Deletes objects pointed to by given vector
void PollEvent(sf::RenderWindow* pTarget, bool* pIsPaused, sf::View* pSimView);                                                         //Call all polled events for the sf::window
void PopulateBodyVectorDisk(std::vector<Body*> *pBodies, float pWidth, float pHeight, float pMaxDist, float pMinDist, float pMinMass, float pMaxMass, float pGalaticCenterMass = 0);  //populate given vector with bodies with given paramiters in a disk formation
void Render(sf::RenderWindow* pTarget, std::vector<Body*> pBodies, sf::Color pObjColor);                                                //Render given body objects to given screen
void SetView(sf::View* pView, sf::RenderWindow* pTarget, float pViewWidth, float pViewHeight);                                          //set the window to the simulation view
void UpdateBodies(std::vector<Body*> &pBodies);                                                                                          //Calculate velocity chance from the bodies force exerted since last update, update position based on velocity, reset force to 0
void DrawNode(Node* pNode, sf::RenderWindow* pTarget);                                                                                  //Draw a node to the screen, and all of its children (recursive)
void CheckNode(Node* pNode, Body* pBody);                                                                                               //Checks if a node is sufficently far away for force calculation, if not recureses on nodes children
void OctreeBodyAttraction();                                                                                                            //Using a calculated oct-tree, calculate the force exerted on each object
void AttractToCenter(std::vector<Body*> &pBodies, float width, float height, float centerMass);                                                                                                                  //Attract each particle to the center of the simulation
void ResetForces(std::vector<Body*> &pBodies);
void RepellFromCenter(std::vector<Body*> &pBodies, float width, float height, float centerMass);

float const DiskRadiusMax = 20000;              //Max and min distances objects will be from the galatic center
float const DiskRadiusMin = 50;
float const GalaticCenterMass = 1000000;        //The mass of the very large object simulating a black hole at the center of a galixy;
float const ObjectMassMax = 10;                  //The max and min mass of the objects in the galixy
float const ObjectMassMin = 1;
float const SimWidth = 327680;                  //Width and height of simulation, needs to be large, particles outside of this range will not be included in the octree
float const SimHeight = 327680;
float const ViewWidth = 1920;                   //Width and height of view of the simulation for the screen. 
float const ViewHeight = 1080;
float const Softener = 10;                      //A softener used for the force calculations, 10 is a good amount
#define pParticlesCount 10000       //Number of particles in simtulation, currently 2^15                                
double const _NODE_THRESHOLD = 0.5;             //Threshold for node calculations   

float zoom = 1;                                 //The current amount of zoom in or out the user has inputed in total
float MouseX = 0;
float MouseY = 0;

std::vector<Body*> Bodies;                      //Container of all Bodies in simulation
Node GlobalNode;
bool IsPaused = false;                          //Contains the state of weather the simulation is paused or not
sf::Color ObjColor(255, 255, 255, 128);         //the defult colour of the objects
sf::View SimulationView;
sf::RenderWindow window(sf::VideoMode(1920, 1080), "N-Body simulation");

int main()
{
	// N Body Code
	try
	{
		PopulateBodyVectorDisk(&Bodies, SimWidth, SimHeight, DiskRadiusMax, DiskRadiusMin, ObjectMassMin, ObjectMassMax, GalaticCenterMass);
	}
	catch (const std::exception &exc)
	{
		std::cerr << exc.what();
	}
	SetView(&SimulationView, &window, ViewWidth, ViewHeight);

	sf::Clock clock;
	float lastTime = 0;
	int frames = 0;
	float avgFps = 0;

	while (window.isOpen())
	{
		PollEvent(&window, &IsPaused, &SimulationView); //These will always be done

		if (!IsPaused)  //These will not if the simulation is paused
		{
			AttractToCenter(Bodies, SimWidth, SimHeight, GalaticCenterMass);
			UpdateBodies(Bodies);
			ResetForces(Bodies);
			GlobalNode.Reset();
			GlobalNode.SetParam(Bodies, SimWidth, SimHeight);
			OctreeBodyAttraction();
		}

		Render(&window, Bodies, ObjColor);
		
		float currentTime = clock.restart().asSeconds();
		float fps = 1.f / currentTime;
		avgFps += fps;
		++frames;
		
		lastTime = currentTime;

	}

	std::cout << "avg: " << avgFps / frames << std::endl;


	DeleteBodies(Bodies);
	// --------------------------------

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaError_t cudaStatus;

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

    return 0;
}

void AttractToCenter(std::vector<Body *> &pBodies, float width, float height, float centerMass)
{
	Body* Temp = new Body(width / 2, height / 2, centerMass); //Create a body at the center of the simulation

	for (unsigned int i = 0; i < pBodies.size(); i++)
	{
		CalculateForce(pBodies[i], Temp, Softener);
	}

	delete Temp;
}

void RepellFromCenter(std::vector<Body*> &pBodies, float width, float height, float centerMass)
{
	Body* Temp = new Body(width / 2, height / 2, centerMass); //Create a body at the center of the simulation

	for (unsigned int i = 0; i < pBodies.size(); i++)
	{
		float vectorx = Temp->posX - pBodies[i]->posX;
		float vectory = Temp->posY - pBodies[i]->posY;

		float distSqr = vectorx * vectorx + vectory * vectory;

		double Dist = (sqrt(distSqr));

		double force = (pBodies[i]->mass * Dist * _GRAV_CONST * 0.0001);

		pBodies[i]->forceX -= vectorx * force;
		pBodies[i]->forceY -= vectory * force;
	}

	delete Temp;
}

void ResetForces(std::vector<Body*> &pBodies)
{
	for (unsigned int i = 0; i < pBodies.size(); i++)
	{
		pBodies[i]->forceX = 0;
		pBodies[i]->forceY = 0;
	}
}

void BodyAttraction(std::vector<Body*> &pBodies, float pSoftener)
{
	for (unsigned int i = 0; i < pBodies.size(); i++)
	{
		for (unsigned int j = 0; j < pBodies.size(); j++)
		{
			CalculateForce(pBodies.at(i), pBodies.at(j), pSoftener); //for each body in pBodies: each other body in pBodies: Calculate attractive force exerted on the first body from the second one
		}
	}
}

void OctreeBodyAttraction()
{

	for (unsigned int i = 0; i < Bodies.size(); i++)
	{
		CheckNode(&GlobalNode, Bodies[i]);
	}
}

inline void CheckNode(Node* pNode, Body* pBody)
{
	if (pNode->Bodies.size() != 0)
	{
		float diffX = (pNode->CenterOfMassx - pBody->posX);
		float diffY = (pNode->CenterOfMassy - pBody->posY);

		float distance = sqrt((diffX) * (diffX)+(diffY) * (diffY));   //Distance from the node to the object                          

		if ((pNode->width / distance) < (_NODE_THRESHOLD) || (pNode->HasChildren == false))     //if sufficently far away or has no children (external node) group node and calculate force
		{
			CalculateForceNode(pBody, pNode, Softener);
			pNode->IsUsed = true;
		}
		else                                                                                    //if not, repeat function with children
		{
			CheckNode(pNode->Child[0], pBody);
			CheckNode(pNode->Child[1], pBody);
			CheckNode(pNode->Child[2], pBody);
			CheckNode(pNode->Child[3], pBody);
		}
	}
}

inline void CalculateForceNode(Body* bi, Node* bj, float pSoftener)  //bi is being attracted to bj. 15 flops of calculation
{
	//vector from the body to the center of mass
	float vectorx = bj->CenterOfMassx - bi->posX;
	float vectory = bj->CenterOfMassy - bi->posY;

	//c^2 = a^2 + b^2 + softener^2
	float distSqr = vectorx * vectorx + vectory * vectory + pSoftener * pSoftener;

	// ivnDistCube = 1/distSqr^(3/2)
	float distSixth = distSqr * distSqr * distSqr;
	double invDistCube = 1.0f / (sqrt(distSixth));


	double force = (bj->TotalMass * bi->mass * invDistCube * _GRAV_CONST);

	bi->forceX += vectorx * force;
	bi->forceY += vectory * force;
}

//__global__ void calculateForceKernel(Body )

inline void CalculateForce(Body* bi, Body* bj, float pSoftener)  //bi is being attracted to bj. 15 flops of calculation
{
	//std::vector from i to j
	float vectorx = bj->posX - bi->posX;
	float vectory = bj->posY - bi->posY;

	//c^2 = a^2 + b^2 + softener^2
	float distSqr = vectorx * vectorx + vectory * vectory + pSoftener * pSoftener;

	// ivnDistCube = 1/distSqr^(3/2)
	float distSixth = distSqr * distSqr * distSqr;
	double invDistCube = 1.0f / (sqrt(distSixth));


	double force = (bj->mass * bi->mass * invDistCube * _GRAV_CONST);

	bi->forceX += vectorx * force;
	bi->forceY += vectory * force;
}

void DeleteBodies(std::vector<Body*> &pBodies)
{
	for (unsigned int i = 0; i < pBodies.size(); i++)
	{
		delete pBodies[i];
	}

	pBodies.clear();
}

void PollEvent(sf::RenderWindow* pTarget, bool* pIsPaused, sf::View* pSimView)
{
	sf::Event event;

	while (pTarget->pollEvent(event))
	{
		if (event.type == sf::Event::Closed)
			pTarget->close();
		if (event.type == sf::Event::KeyPressed)
		{
			if (event.key.code == sf::Keyboard::Space)
				*pIsPaused = !*pIsPaused;                   //toggle what is pointed to by IsPaused
		}
		if (event.type == sf::Event::MouseWheelScrolled)
		{
			zoom *= 1 + (static_cast <float> (-event.mouseWheelScroll.delta) / 10); //for each notch down, -10%, for each notch up, +10%
			pSimView->zoom(1 + (static_cast <float> (-event.mouseWheelScroll.delta) / 10));
		}
	}

	if (sf::Mouse::getPosition().x > (1920 - 20))
		SimulationView.move(2 * zoom, 0);
	if (sf::Mouse::getPosition().x < (0 + 20))
		SimulationView.move(-2 * zoom, 0);
	if (sf::Mouse::getPosition().y >(1080 - 20))
		SimulationView.move(0, 2 * zoom);
	if (sf::Mouse::getPosition().y < (0 + 20))
		SimulationView.move(0, -2 * zoom);

	pTarget->setView(*pSimView);
}

__global__ void setup_curand(curandState_t *states)
{
	unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	curand_init(0, idx, 0, &states[idx]);
}

__global__ void calculateMass(curandState_t *states, float *minMass, float *maxMass, float *varsArray)
{
	unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	varsArray[threadIdx.x] = *minMass + (curand_uniform(&states[idx]) * (*maxMass));
}

__global__ void calculatePosX(curandState_t *states, float *minDist, float *maxDist, float *width, float *varsArray)
{
	unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	float theta = curand_uniform(&states[idx]) * 2 * 3.14159265;
	float distanceCoefficient = curand_uniform(&states[idx]);

	float distance = *minDist + ((*maxDist - *minDist) * (distanceCoefficient * distanceCoefficient));

	varsArray[idx] = cos(theta) * distance + (*width / 2);
}

__global__ void calculatePosY(curandState_t *states, float *minDist, float *maxDist, float *height, float *varsArray)
{
	unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	float theta = curand_uniform(&states[idx]) * 2 * 3.14159265;
	float distanceCoefficient = curand_uniform(&states[idx]);

	float distance = *minDist + ((*maxDist - *minDist) * (distanceCoefficient * distanceCoefficient));

	varsArray[idx] = sin(theta) * distance + (*height / 2);
}

__global__ void calculateVelocityX(curandState_t *states, float *minDist, float *maxDist, float *galacticCenterMass, float *varsArray)
{
	unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	float theta = curand_uniform(&states[idx]) * 2 * 3.14159265;
	float distanceCoefficient = curand_uniform(&states[idx]);

	float distance = *minDist + ((*maxDist - *minDist) * (distanceCoefficient * distanceCoefficient));
	float orbitalVelocity = sqrt((*galacticCenterMass * static_cast <float> (_GRAV_CONST)) / distance);                  //Calculate the orbital velocity required to orbit the galatic centre   

	varsArray[idx] = (sin(theta) * orbitalVelocity);
}

__global__ void calculateVelocityY(curandState_t *states, float *minDist, float *maxDist, float *galacticCenterMass, float *varsArray)
{
	unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	float theta = curand_uniform(&states[idx]) * 2 * 3.14159265;
	float distanceCoefficient = curand_uniform(&states[idx]);

	float distance = *minDist + ((*maxDist - *minDist) * (distanceCoefficient * distanceCoefficient));
	float orbitalVelocity = sqrt((*galacticCenterMass * static_cast <float> (_GRAV_CONST)) / distance);                  //Calculate the orbital velocity required to orbit the galatic centre   

	varsArray[idx] = (-cos(theta) * orbitalVelocity);
}

void PopulateBodyVectorDisk(std::vector<Body*> *pBodies, float pWidth, float pHeight, float pMaxDist, float pMinDist, float pMinMass, float pMaxMass, float pGalaticCenterMass)
{
	auto start = std::chrono::system_clock::now();

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dist(0, 1); //pMinDist, pMaxDist);
	std::uniform_real_distribution<float> angle(0, 2*_PI);
	std::uniform_real_distribution<float> mass(pMinMass, pMaxMass);
	

	// cuda stuff
	float *massArray;
	float *posXArray;
	float *posYArray;
	float *velocityXArray;
	float *velocityYArray;
	float *width;
	float *height;
	float *minMass;
	float *maxMass;
	float *minDist;
	float *maxDist;
	float *galacticCenterMass;

	curandState_t *states;

	cudaError cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}

	cudaStatus = cudaMalloc((void**)&states, sizeof(curandState_t) * pParticlesCount);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc((void**)&width, sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
	cudaStatus = cudaMalloc((void**)&height, sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
	cudaStatus = cudaMalloc((void**)&minMass, sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
	cudaStatus = cudaMalloc((void**)&maxMass, sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
	cudaStatus = cudaMalloc((void**)&minDist, sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
	cudaStatus = cudaMalloc((void**)&maxDist, sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
	cudaStatus = cudaMalloc((void**)&galacticCenterMass, sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
	cudaStatus = cudaMalloc((void**)&massArray, sizeof(float) * pParticlesCount);	// this array will contain all the values needed to create all the particles
																// it is multiplied by 5 because there will be 5 variables for each Body
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc((void**)&posXArray, sizeof(float) * pParticlesCount);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc((void**)&posYArray, sizeof(float) * pParticlesCount);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc((void**)&velocityXArray, sizeof(float) * pParticlesCount);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc((void**)&velocityYArray, sizeof(float) * pParticlesCount);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	cudaStatus = cudaMemcpy(width, &pWidth, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy width failed!");
	}
	cudaStatus = cudaMemcpy(height, &pHeight, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy height failed!");
	}
	cudaStatus = cudaMemcpy(minMass, &pMinMass, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy minMass failed!");
	}
	cudaStatus = cudaMemcpy(maxMass, &pMaxMass, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy maxMass failed!");
	}
	cudaStatus = cudaMemcpy(minDist, &pMinDist, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy minDist failed!");
	}
	cudaStatus = cudaMemcpy(maxDist, &pMaxDist, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy maxDist failed!");
	}
	cudaStatus = cudaMemcpy(galacticCenterMass, &pGalaticCenterMass, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy galactic failed!");
	}

	setup_curand <<<pParticlesCount/1024, 1024 >>> (states);

	calculateMass <<<pParticlesCount/1024,1024>>> (states, minMass, maxMass, massArray);
	calculatePosX <<<pParticlesCount / 1024, 1024 >>> (states, minDist, maxDist, width, posXArray);
	calculatePosY <<<pParticlesCount / 1024, 1024 >>> (states, minDist, maxDist, height, posYArray);
	calculateVelocityX <<<pParticlesCount / 1024, 1024 >>> (states, minDist, maxDist, galacticCenterMass, velocityXArray);
	calculateVelocityY <<<pParticlesCount / 1024, 1024 >>> (states, minDist, maxDist, galacticCenterMass, velocityYArray);

	std::vector<float> massResults(pParticlesCount);
	std::vector<float> posXResults(pParticlesCount);
	std::vector<float> posYResults(pParticlesCount);
	std::vector<float> velocityXResults(pParticlesCount);
	std::vector<float> velocityYResults(pParticlesCount);

	cudaMemcpy(&massResults[0], massArray, sizeof(float) * pParticlesCount, cudaMemcpyDeviceToHost);
	cudaMemcpy(&posXResults[0], posXArray, sizeof(float) * pParticlesCount, cudaMemcpyDeviceToHost);
	cudaMemcpy(&posYResults[0], posYArray, sizeof(float) * pParticlesCount, cudaMemcpyDeviceToHost);
	cudaMemcpy(&velocityXResults[0], velocityXArray, sizeof(float) * pParticlesCount, cudaMemcpyDeviceToHost);
	cudaMemcpy(&velocityYResults[0], velocityYArray, sizeof(float) * pParticlesCount, cudaMemcpyDeviceToHost);

	// create the bodies
	for (int i = 0; i < pParticlesCount; i++) {
		//std::cout << posXResults[i] << " " << posYResults[i] << " " << massResults[i] << " " << velocityXResults[i] << " " << velocityYResults[i] << std::endl;
		float theta = angle(gen);
		float distanceCoefficient = dist(gen);

		float distance = pMinDist + ((pMaxDist - pMinDist) * (distanceCoefficient * distanceCoefficient));                    //Distance point will be from the galatic center, between MinDiskRadius and MaxDiskRadius

		float positionx = cos(theta) * distance + (pWidth / 2);                                                             //set positionx and positiony to be the point you get when you go in the direction of 'angle' till you have traveled 'distance' 
		float positiony = sin(theta) * distance + (pHeight / 2);

		float orbitalVelocity = sqrt((pGalaticCenterMass * static_cast <float> (_GRAV_CONST)) / distance);                  //Calculate the orbital velocity required to orbit the galatic centre   

		float velocityx = (sin(theta) * orbitalVelocity);
		float velocityy = (-cos(theta) * orbitalVelocity);

		//std::cout << positionx << " " << positiony << " " << mass(gen) << " " << velocityx << " " << velocityy << std::endl;

		pBodies->push_back(new Body(positionx, positiony, mass(gen), velocityx, velocityy));
		//pBodies->push_back(new Body(posXResults[i], posYResults[i], mass(gen)));
	}

	cudaFree(massArray);
	cudaFree(posXArray);
	cudaFree(posYArray);
	cudaFree(velocityXArray);
	cudaFree(velocityYArray);
	cudaFree(width);
	cudaFree(height);
	cudaFree(minMass);
	cudaFree(maxMass);
	cudaFree(minDist);
	cudaFree(maxDist);
	cudaFree(galacticCenterMass);
	cudaFree(states);

	auto end = std::chrono::system_clock::now();
	auto total = end - start;
	std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(total).count() << std::endl;
}


void Render(sf::RenderWindow* pTarget, std::vector<Body*> pBodies, sf::Color pObjColor)
{
	pTarget->clear();

	sf::RectangleShape Temp;
	//Temp.setFillColor(pObjColor);

	for (unsigned int i = 0; i < pBodies.size(); i++)
	{
		if (zoom > 1)
			Temp.setSize(sf::Vector2f(pBodies.at(i)->mass * zoom, pBodies.at(i)->mass * zoom));
		else
			Temp.setSize(sf::Vector2f(pBodies.at(i)->mass, pBodies.at(i)->mass));

		float ForceCoiffecent = sqrt(pBodies.at(i)->forceX * pBodies.at(i)->forceX + pBodies.at(i)->forceY * pBodies.at(i)->forceY) * (40000 * _GRAV_CONST);

		if (ForceCoiffecent > 1)
			ForceCoiffecent = 1;

		float Red, Green, Blue;

		Blue = 1 - (ForceCoiffecent);

		if (ForceCoiffecent < 0.2)
			Red = (ForceCoiffecent) * 5;
		else
			Red = 1;

		if (ForceCoiffecent < 0.5)
			Green = (ForceCoiffecent) * 2;
		else
			Green = 1;

		Temp.setFillColor(sf::Color(Red * 255, Green * 255, Blue * 255, 128));
		Temp.setPosition(pBodies.at(i)->posX, pBodies.at(i)->posY);
		pTarget->draw(Temp);
	}

	//DrawNode(&GlobalNode, pTarget);

	pTarget->display();
}

void DrawNode(Node* pNode, sf::RenderWindow* pTarget)
{
	sf::RectangleShape Temp;
	Temp.setFillColor(sf::Color(0, 0, 0, 0));
	Temp.setOutlineThickness(zoom);
	Temp.setOutlineColor(sf::Color(0, 255, 0, 16));
	Temp.setPosition(pNode->posX, pNode->posY);
	Temp.setSize(sf::Vector2f(pNode->width, pNode->height));

	pTarget->draw(Temp);
	if (pNode->HasChildren) //recercivly draw all children
	{
		DrawNode(pNode->Child[0], pTarget);
		DrawNode(pNode->Child[1], pTarget);
		DrawNode(pNode->Child[2], pTarget);
		DrawNode(pNode->Child[3], pTarget);
	}
}

void SetView(sf::View* pView, sf::RenderWindow* pTarget, float pViewWidth, float pViewHeight)
{
	pView->reset(sf::FloatRect(SimWidth / 2 - pViewWidth / 2, SimHeight / 2 - pViewHeight / 2, pViewWidth, pViewHeight));
	pView->setViewport(sf::FloatRect(0.f, 0.f, 1.f, 1.f));
	pTarget->setView(*pView);
}


void UpdateBodies(std::vector<Body*> &pBodies)
{

	for (unsigned int i = 0; i < pBodies.size(); i++)
	{
		if ((pBodies[i]->posX > SimWidth && pBodies[i]->velX > 0) || (pBodies[i]->posX < 0 && pBodies[i]->velX < 0))
		{
			pBodies[i]->velX = -pBodies[i]->velX;
		}

		if ((pBodies[i]->posY > SimHeight && pBodies[i]->velY > 0) || (pBodies[i]->posY < 0 && pBodies[i]->velY < 0))
		{
			pBodies[i]->velY = -pBodies[i]->velY;
		}

		pBodies.at(i)->velX += pBodies.at(i)->forceX / pBodies.at(i)->mass;     //f = ma => force = mass * accelleration. Therefor
		pBodies.at(i)->velY += pBodies.at(i)->forceY / pBodies.at(i)->mass;     //a = f/m => accelleration = force / mass

		pBodies.at(i)->posX += pBodies.at(i)->velX;
		pBodies.at(i)->posY += pBodies.at(i)->velY;
	}
}