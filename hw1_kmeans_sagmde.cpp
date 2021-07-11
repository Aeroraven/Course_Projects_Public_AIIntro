#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/imgproc/types_c.h"
#include <iostream>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <string>
#include <istream>
#include <sstream>
#include <fstream>
#include <exception>
using namespace cv;
using namespace std;

namespace Clustering {
	struct Coord2d {
		double x;
		double y;
		bool operator == (const Coord2d& p) const {
			return x == p.x && y == p.y;
		}
	};
	struct Coord3b {
		int x;
		int y;
		int z;
	};
	class KException_IrrVector :public exception {
		const char* what() const noexcept {
			return "KException: Operating vectors with different dimensions";
		}
	};
	
	
	class KVector {
	public:
		double* x = nullptr;
		int dimension = 0;
		KVector(const KVector& p) {
			PasteFrom(p);
		}
		KVector() {
			return;
		}
		~KVector() {
			if (x != nullptr) delete[]x;
		}
		const KVector& operator = (const KVector& p) {
			PasteFrom(p);
			return *this;
		}
		void Clear() {
			for (int i = 0; i < dimension; i++)x[i] = 0;
		}
		void PasteFrom(const KVector& p) {
			if (x != nullptr) delete[]x;
			x = new double[p.dimension];
			dimension = p.dimension;
			for (int i = 0; i < dimension; i++)x[i] = p.x[i];
		}
		void PasteFrom(int array_limit, double* array_list) {
			if (x != nullptr) delete[]x;
			x = new double[array_limit];
			dimension = array_limit;
			for (int i = 0; i < array_limit; i++)x[i] = array_list[i];
		}
		void PasteFrom(int array_limit, vector<double> array_list) {
			if (x != nullptr) delete[]x;
			x = new double[array_limit];
			for (int i = 0; i < array_limit; i++)x[i] = array_list[i];
			dimension = array_limit;

		}
		void AllocateNew(int array_limit) {
			if (x != nullptr) delete[]x;
			x = new double[array_limit];
			dimension = array_limit;
		}
		bool operator == (const KVector& p) const {
			if (dimension != p.dimension)return false;
			for (int i = 0; i < dimension; i++) {
				if (x[i] != p.x[i])return false;
			}
			return true;
		}
	};

	class RandomSeed {
	public:
		RandomSeed() {
			srand(time(NULL));
		}
	};
	class Random {
	public:
		const double pi = 3.14159;
		Random() {
			
		}
		int RandomInt(int l, int r) {
			return rand() % (r - l + 1) + l;
		}
		double StdUniformDistribution() {
			return static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
		}
		double UniformDistribution(double l, double r) {
			return StdUniformDistribution() * (r - l) + l;
		}
		double StdNormalDistribution() {
			double u1 = StdUniformDistribution(), u2 = StdUniformDistribution();
			double z1 = sqrt(-2.0 * log(u1)) * sin(2 * pi * u2);
			return z1;
		}
		Coord2d StdIrrNormalDistribution() {
			double u1 = StdUniformDistribution(), u2 = StdUniformDistribution();
			double z1 = sqrt(-2.0 * log(u1)) * sin(2 * pi * u2);
			double z2 = sqrt(-2.0 * log(u1)) * cos(2 * pi * u2);
			return { z1,z2 };
		}
		Coord2d IrrNormalDistribution(double m_1, double d_1, double m_2, double d_2) {
			double u1 = StdUniformDistribution(), u2 = StdUniformDistribution();
			double z1 = sqrt(-2.0 * log(u1)) * sin(2 * pi * u2);
			double z2 = sqrt(-2.0 * log(u1)) * cos(2 * pi * u2);
			z1 = (z1 * sqrt(d_1)) + m_1;
			z2 = (z2 * sqrt(d_2)) + m_2;
			return { z1,z2 };
		}
	};
	class KMeansSolution {
	public:
		vector<KVector> pointlist;
		vector<int> pointlist_tag;
		vector<int> cluster_result;
		vector<KVector> cluster_center;
		vector<int> cluster_dominance;
		vector<double> cluster_accuracy;

		template<class T> T Min(const T& a, const T& b)const {
			return (a > b) ? b : a;
		}
		template<class T> T Max(const T& a, const T& b)const {
			return (a < b) ? b : a;
		}
		void CalculateDominance(int tagTypes=10) {
			cluster_dominance.clear();
			//cout << endl<<"DOMINANT CLUSTER:";
			for (int i = 0; i < cluster_center.size(); i++) {
				cluster_dominance.push_back(0);
				vector<int> cntw;
				for (int j = 0; j < tagTypes; j++) {
					cntw.push_back(0);
				}
				for (int j = 0; j < pointlist.size(); j++) {
					if (cluster_result[j] == i) {
						cntw[pointlist_tag[j]]++;
					}
				}
				int mv = 0;
				for (int j = 0; j < tagTypes; j++) {
					if (cntw[mv] < cntw[j])mv = j;
				}
				cluster_dominance[i] = mv;
				//cout << mv << ",";
			}
			//cout << endl;
		}
		double CalculateAccuracy() {
			return 0;
			CalculateDominance();
			int correct = 0;
			for (int i = 0; i < pointlist.size(); i++) {
				if (pointlist_tag[i] == cluster_dominance[cluster_result[i]]) {
					correct ++;
				}
			}
			return (double)correct / pointlist.size();
		}
		double Dist(KVector x, KVector y) {
			//if (x.dimension != y.dimension)throw KException_IrrVector();
			double dist = 0;
			for (int i = 0; i < x.dimension; i++) {
				dist += (x.x[i] - y.x[i]) * (x.x[i] - y.x[i]);
			}
			return sqrt(dist);
		}
		void PasteFrom(const KMeansSolution& p) {
			pointlist.clear();
			cluster_result.clear();
			cluster_center.clear();
			pointlist_tag.clear();
			for (int i = 0; i < p.pointlist.size(); i++)pointlist.push_back(p.pointlist[i]);
			for (int i = 0; i < p.cluster_center.size(); i++)cluster_center.push_back(p.cluster_center[i]);
			for (int i = 0; i < p.cluster_result.size(); i++)cluster_result.push_back(p.cluster_result[i]);
			for (int i = 0; i < p.pointlist_tag.size(); i++)pointlist_tag.push_back(p.pointlist_tag[i]);
		}
		vector<double> GetClusterDistortion() {
			vector<double> cluster_single_distortion;
			vector<int> cluster_member_quantity;
			for (int i = 0; i < cluster_center.size(); i++) {
				cluster_single_distortion.push_back(0.0);
				cluster_member_quantity.push_back(0);
			}
			for (int i = 0; i < pointlist.size(); i++) {
				cluster_single_distortion[cluster_result[i]] += Dist(pointlist[i], cluster_center[cluster_result[i]]);
				cluster_member_quantity[cluster_result[i]]++;
			}
			return cluster_single_distortion;
		}
		vector<double> GetClusterUtility() {
			vector<double> cluster_distortion = GetClusterDistortion();
			double tmp = 0.0;
			for (int i = 0; i < cluster_distortion.size(); i++) {
				tmp += cluster_distortion[i];
			}
			tmp /= static_cast<double> (cluster_distortion.size());
			for (int i = 0; i < cluster_distortion.size(); i++) {
				cluster_distortion[i] /= tmp;
			}
			return cluster_distortion;
		}
		vector<KVector> GetUtilityHyperbox(int cluster_id) { //This function only returns 2-element vector containing both sides of the diagonal
			KVector lt_point, rt_point;
			lt_point.AllocateNew(pointlist[0].dimension);
			rt_point.AllocateNew(pointlist[0].dimension);
			for (int i = 0; i < lt_point.dimension; i++) {
				lt_point.x[i] = 1e+100;
				rt_point.x[i] = -1e+100;
			}
			for (int i = 0; i < pointlist.size(); i++) {
				if (cluster_result[i] == cluster_id) {
					for (int j = 0; j < lt_point.dimension; j++) {
						lt_point.x[j] = Min(lt_point.x[j], pointlist[i].x[j]);
						rt_point.x[j] = Max(rt_point.x[j], pointlist[i].x[j]);
					}
				}
			}
			return{ lt_point,rt_point };
		}
		void ReadjustDistCenter(int clu_low, int clu_high) {
			vector<KVector> diag_point = GetUtilityHyperbox(clu_high);
			vector<KVector> readjusted_center;
			KVector diag_vector;
			diag_vector.AllocateNew(diag_point[0].dimension);
			for (int i = 0; i < diag_vector.dimension; i++) {
				diag_vector.x[i] = diag_point[0].x[i] - diag_point[1].x[i];
			}

			readjusted_center.push_back(diag_point[1]);
			readjusted_center.push_back(diag_point[1]);
			for (int i = 0; i < diag_point[0].dimension; i++) {
				readjusted_center[0].x[i] += 1.0 / 3 * diag_vector.x[i];
				readjusted_center[1].x[i] += 2.0 / 3 * diag_vector.x[i];
			}
			cluster_center[clu_low].PasteFrom(readjusted_center[0]);
			cluster_center[clu_high].PasteFrom(readjusted_center[1]);
		}

		void OptimizeCenter(int iteration_times,bool reallocate=true) {
			while (iteration_times--) {
				double presse = GetSSE(), newsse = 0;
				for (int i = 0; i < pointlist.size(); i++) {
					int best_cluster_id = -1;
					double best_cluster_distance = 1e+100;
					for (int j = 0; j < cluster_center.size(); j++) {
						double dist = Dist(pointlist[i], cluster_center[j]);
						if (dist < best_cluster_distance) {
							best_cluster_id = j;
							best_cluster_distance = dist;
						}
					}
					cluster_result[i] = best_cluster_id;
				}
				if (reallocate == false)return;
				newsse = GetSSE();
				if (fabs(newsse - presse) < 1e-5)return;
				KVector zero_vector;
				zero_vector.AllocateNew(cluster_center[0].dimension);
				for (int i = 0; i < cluster_center[0].dimension; i++) {
					zero_vector.x[i] = 0;
				}
				vector<KVector> new_cluster_center;
				vector<int> cluster_member_counter;
				for (int i = 0; i < cluster_center.size(); i++) {
					new_cluster_center.push_back(zero_vector);
					cluster_member_counter.push_back(0);
				}
				for (int i = 0; i < pointlist.size(); i++) {
					for (int j = 0; j < pointlist[i].dimension; j++) {
						new_cluster_center[cluster_result[i]].x[j] += pointlist[i].x[j];
						
					}
					cluster_member_counter[cluster_result[i]]++;
					

				}
				for (int i = 0; i < cluster_center.size(); i++) {
					for (int j = 0; j < cluster_center[i].dimension; j++) {
						cluster_center[i].x[j] = new_cluster_center[i].x[j] / cluster_member_counter[i];
					}
				}
			}
		}
		void ApplyNormalDisturbance(int cluster_id,double normal_sqdev = 1.0,double scale_factor = 1.0) {
			Random R;
			for (int i = 0; i < cluster_center[0].dimension; i++) {
				double disturb_x = R.StdNormalDistribution() * sqrt(normal_sqdev) * scale_factor;
				cluster_center[cluster_id].x[i] += disturb_x;
			}
		}
		void GenerateGaussianPerturbedSol(double normal_sqdev = 1.0, double scale_factor = 5.0) {
			for (int i = 0; i < cluster_center.size(); i++) {
				ApplyNormalDisturbance(i, normal_sqdev, scale_factor);
			}
			OptimizeCenter(2,true);
			return;
		}
		void GenerateDistortionSol() {
			// This change equalizes all clusters' sizes.
			vector<double> cluster_utility = GetClusterUtility();
			vector<int> clow_list;
			int culow = 0, cuhigh = 0;
			double culow_v = 1e100, cuhigh_v = -1e100;
			for (int i = 0; i < cluster_center.size(); i++) {
				if (cluster_utility[i] < culow_v) {
					culow_v = cluster_utility[i];
					culow = i;
				}
				if (cluster_utility[i] > cuhigh_v) {
					cuhigh_v = cluster_utility[i];
					cuhigh = i;
				}
				if (cluster_utility[i] < 1) {
					clow_list.push_back(i);
				}
			}
			
			if (clow_list.size() >= 1) {
				Random R;
				culow = clow_list[R.RandomInt(0, clow_list.size() - 1)];
			}
			
			ReadjustDistCenter(culow, cuhigh);
			OptimizeCenter(15);
		}
		double GetSSE() {
			double errs = 0.0;
			double tmp;
			vector<int> cluster_member_quantity;
			for (int i = 0; i < cluster_center.size(); i++) {
				cluster_member_quantity.push_back(0);
			}
			for (int i = 0; i < pointlist.size(); i++) {
				tmp = Dist(pointlist[i], cluster_center[cluster_result[i]]);
				cluster_member_quantity[cluster_result[i]]++;
				errs += tmp * tmp;
			}
			//If an inappropriate shift of center emerges, points might be classified to other clusters
			//Steps below avoids accepting the improper shifts of centers and avoids clusters containing no points
			for (int i = 0; i < cluster_center.size(); i++) {
				if (cluster_member_quantity[i] == 0) {
					errs = 1e+100; 
				}
			}
			return errs;
		}
	};
	class KMeans {
	public:
		int kval;
		vector<KVector> pointlist;
		vector<int> cluster_result;
		vector<KVector> cluster_center;
		vector<int> id_list;
		vector<int> pointlist_tag;
		vector<int>cluster_dominance;
		Random R;
		double Dist(KVector x, KVector y) {
			if (x.dimension != y.dimension)throw KException_IrrVector();
			double dist = 0;
			for (int i = 0; i < x.dimension; i++) {
				dist += (x.x[i] - y.x[i]) * (x.x[i] - y.x[i]);
			}
			return sqrt(dist);
		}
		void SetK(int _k) {
			kval = _k;
		}
		void LoadSamples(const vector<KVector>& _samples) {
			int sz = _samples.size();
			for (int i = 0; i < sz; i++) {
				pointlist.push_back(_samples[i]);
				cluster_result.push_back(-1);
				id_list.push_back(id_list.size());
			}
		}
		void LoadSamplesTag(const vector<int>& _samples) {
			int sz = _samples.size();
			for (int i = 0; i < sz; i++) {
				pointlist_tag.push_back(_samples[i]);
				cout << "TAG:"<<_samples[i] <<","<< _samples.size()<< endl;
			}
		}
		void LoadSamples(const KVector& _sample) {
			pointlist.push_back(_sample);
			cluster_result.push_back(-1);
			id_list.push_back(id_list.size());
		}

		double CalculateSSE() {
			
			double sse = 0.0;
			for (int i = 0; i < cluster_center.size(); i++) {
				double s = 0.0;
				for (int j = 0; j < pointlist.size(); j++) {
					if (cluster_result[j] == i) {
						double dst = Dist(pointlist[j], cluster_center[i]);
						s += dst * dst;
					}
				}
				sse += s;
			}
			return sse;
		}
		void LoadFromSolution(KMeansSolution &p) {
			pointlist.clear();
			cluster_result.clear();
			cluster_center.clear();
			for (int i = 0; i < p.cluster_center.size(); i++)cluster_center.push_back(p.cluster_center[i]);
			for (int i = 0; i < p.pointlist.size(); i++)pointlist.push_back(p.pointlist[i]);
			for (int i = 0; i < p.cluster_result.size(); i++)cluster_result.push_back(p.cluster_result[i]);
		}
		void OptimalRun(KMeansSolution &ret_val, int iteration_count, int max_iteration=45, double cooldown_rate=0.95,double initial_temp = 10000,double initial_temp_distort = 10000, double terminal_temp = 100, double terminal_temp_distort = 10,double scale_factor=0.0005,int de_interval=3) {
			Run(iteration_count);
			KMeansSolution initial_sol, adjacent_sol, adjacent_best_sol, current_sol, optimal_sol;
			for (int i = 0; i < pointlist.size(); i++)initial_sol.pointlist.push_back(pointlist[i]);
			for (int i = 0; i < cluster_center.size(); i++)initial_sol.cluster_center.push_back(cluster_center[i]);
			for (int i = 0; i < cluster_result.size(); i++)initial_sol.cluster_result.push_back(cluster_result[i]);
			for (int i = 0; i < pointlist_tag.size(); i++)initial_sol.pointlist_tag.push_back(pointlist_tag[i]);;
			double temp = initial_temp, temp_distort = initial_temp_distort;
			double temp_best = initial_temp;
			current_sol.PasteFrom(initial_sol);
			optimal_sol.PasteFrom(initial_sol);
			while (temp > terminal_temp) {
				vector<int> cluster_member_quantity;
				int sum = 0;
				for (int i = 0; i < current_sol.cluster_center.size(); i++) {
					cluster_member_quantity.push_back(0);
				}
				for (int i = 0; i < current_sol.pointlist.size(); i++) {
					cluster_member_quantity[current_sol.cluster_result[i]]++;
				}

				cout << "SA_1: T=" << temp << ", E=" << current_sol.GetSSE() << " CS:";
				for (int i = 0; i < current_sol.cluster_center.size(); i++) {
					cout << cluster_member_quantity[i] << ",";
					sum += cluster_member_quantity[i];
				}
				cout << ", SUM:" << sum << ", ACCURACY:" << current_sol.CalculateAccuracy() << endl;
				for (int mi = 0; mi < max_iteration; mi++) {
					//Update the optimal solution
					if (current_sol.GetSSE() < optimal_sol.GetSSE()) {
						optimal_sol.PasteFrom(current_sol);
						temp_best = temp;
					}
					//Attempt to evaluate the neighbor solution with Gaussian perturbation
					adjacent_sol.PasteFrom(current_sol);
					adjacent_sol.GenerateGaussianPerturbedSol(1.0,scale_factor*sqrt(temp)/10);
					double adjacent_sse = adjacent_sol.GetSSE();
					double current_sse = current_sol.GetSSE();
					if (adjacent_sse<current_sse || exp((current_sse - adjacent_sse) / temp)>R.StdUniformDistribution()) {
						current_sol.PasteFrom(adjacent_sol);
						current_sse = adjacent_sse;
					}
					
					//Attempt to evaluate the neighbor solution with Distortion equalization
					if (mi % de_interval == 0) {
						adjacent_sol.PasteFrom(current_sol);
						adjacent_sol.GenerateDistortionSol();
						adjacent_sse = adjacent_sol.GetSSE();
						if (adjacent_sse<current_sse || exp((current_sse - adjacent_sse) / temp_distort)>R.StdUniformDistribution()) {
							current_sol.PasteFrom(adjacent_sol);
							current_sse = adjacent_sse;
						}
					}
				}
				temp *= cooldown_rate;
				temp_distort *= cooldown_rate;
			}
			temp = temp_best;
			current_sol.PasteFrom(optimal_sol);
			while (temp > terminal_temp) {
				vector<int> cluster_member_quantity;
				for (int i = 0; i < current_sol.cluster_center.size(); i++) {
					cluster_member_quantity.push_back(0);
				}
				for (int i = 0; i < current_sol.pointlist.size(); i++) {
					cluster_member_quantity[current_sol.cluster_result[i]]++;
				}

				if (current_sol.GetSSE() < optimal_sol.GetSSE()) {
					optimal_sol.PasteFrom(current_sol);
					temp_best = temp;
				}
				cout << "SA_2: T=" << temp << ", E=" << current_sol.GetSSE() << " CS:";
				for (int i = 0; i < current_sol.cluster_center.size(); i++) {
					cout << cluster_member_quantity[i] << ",";
				}
				cout <<" ACCURACY:" << current_sol.CalculateAccuracy() << endl;
				for (int i = 0; i < max_iteration; i++) {
					adjacent_sol.PasteFrom(current_sol);
					adjacent_sol.GenerateGaussianPerturbedSol(1.0, scale_factor * sqrt(temp) / 10);
					double adjacent_sse = adjacent_sol.GetSSE();
					double current_sse = current_sol.GetSSE();
					if (adjacent_sse<current_sse || exp((current_sse - adjacent_sse) / temp)>R.StdUniformDistribution()) {
						current_sol.PasteFrom(adjacent_sol);
						current_sse = adjacent_sse;
					}
				}
				temp *= cooldown_rate;
			}
			
			ret_val.PasteFrom(optimal_sol);
		}
		void CalculateDominance(int tagTypes = 10) {
			cluster_dominance.clear();

			for (int i = 0; i < cluster_center.size(); i++) {
				cluster_dominance.push_back(0);
				vector<int> cntw;
				for (int j = 0; j < tagTypes; j++) {
					cntw.push_back(0);
				}
				for (int j = 0; j < pointlist.size(); j++) {
					if (cluster_result[j] == i) {
						cntw[pointlist_tag[j]]++;
					}
				}
				int mv = 0;
				for (int j = 0; j < tagTypes; j++) {
					if (cntw[mv] < cntw[j])mv = j;
				}
				cluster_dominance[i] = mv;
			}
		}
		double CalculateAccuracy() {
			return 0;
			CalculateDominance();
			int correct = 0;
			for (int i = 0; i < pointlist.size(); i++) {
				if (pointlist_tag[i] == cluster_dominance[cluster_result[i]]) {
					correct++;
				}
			}
			return (double)correct / pointlist.size();
		}
		int Run(int iteration_count) {
			for (int i = 0; i < kval; i++) {
				//int temp = R.RandomInt(0, id_list.size());
				int temp = 1;
				cluster_center.push_back(pointlist[id_list[temp]]);
				auto it = find(id_list.begin(), id_list.end(), id_list[temp]);
				if (it != id_list.end()) {
					id_list.erase(it);
				}
				else {
					cout << "Failed To Choose Starting Points" << endl;
					return -1;
				}
			}
			for (int T = 0; T < iteration_count; T++) {
				//Calculate Cluster Categories
				for (int i = 0; i < pointlist.size(); i++) {
					int best_cluster_id = -1;
					double best_cluster_distance = 1e+100;
					for (int j = 0; j < cluster_center.size(); j++) {
						if (Dist(pointlist[i], cluster_center[j]) < best_cluster_distance) {
							best_cluster_distance = Dist(pointlist[i], cluster_center[j]);
							best_cluster_id = j;
						}
					}
					cluster_result[i] = best_cluster_id;
				}
				//Calculate Center
				
				for (int i = 0; i < cluster_center.size(); i++) {
					KVector newcenter;
					newcenter.AllocateNew(cluster_center[0].dimension);
					newcenter.Clear();
					int cnt = 0;
					for (int j = 0; j < pointlist.size(); j++) {
						if (cluster_result[j] == i) {
							cnt++;
							for (int k = 0; k < pointlist[0].dimension; k++) {
								newcenter.x[k] += pointlist[j].x[k];
							}
							
						}
					}
					for (int k = 0; k < pointlist[0].dimension; k++) {
						newcenter.x[k] /= static_cast<double>(cnt);
					}

					cluster_center[i].PasteFrom(newcenter);
				}
				//Calculate Error
				//cout << "Iteration Epoch " << T << ": Err=" << CalculateSSE() << endl;
			}
			vector<int> cluster_member_quantity;
			for (int i = 0; i < cluster_center.size(); i++) {
				cluster_member_quantity.push_back(0);
			}
			for (int i = 0; i < pointlist.size(); i++) {
				cluster_member_quantity[cluster_result[i]]++;
			}
			cout << "Initial Sol " << ": E=" << CalculateSSE() << ",CS=";
			for (int i = 0; i < cluster_center.size(); i++) {
				cout << cluster_member_quantity[i] << ",";
			}
			cout << "ACCURACY:" << CalculateAccuracy() << endl;
			return 0;
		}

	};

	class KMeansDemonstrator {
	public:
		KMeans km;
		KMeans kmOpt;
		
		vector<Coord3b> color_preset = { {255,100,100},{0,255,0},{0,0,255},{0,255,255},{255,0,255},{255,255,0},{255,255,255},{128,128,128} };

		void iris_cluster() {
			wine_cluster(4, 3, "C:\\a\\iris.csv");

			return;
			vector<vector<double>> iris_parameters;
			vector<KVector> iris_vector;
			vector<int> iris_spec;

			//Load from file
			ifstream iris_dataset_file;
			iris_dataset_file.open("C:\\a\\iris.csv");
			
			while (!iris_dataset_file.eof()) {
				string file_line;
				stringstream csv_reader;
				iris_dataset_file >> file_line;
				for (int i = 0; i < file_line.size(); i++) {
					if (file_line[i] == ',')file_line[i] = ' ';
				}
				csv_reader << file_line;
				double temp;
				int temp2;
				iris_parameters.push_back(vector<double>{});
				iris_spec.push_back(0);
				KVector temp_vec;
				temp_vec.AllocateNew(4);
				temp_vec.Clear();
				iris_vector.push_back(temp_vec);
				cout << "IRIS:";
				for (int i = 0; i < 4; i++) {
					csv_reader >> temp;
					iris_parameters[iris_parameters.size() - 1].push_back(temp);
					iris_vector[iris_vector.size() - 1].x[i] = temp;
					cout << temp << ",";
				}
				csv_reader >> iris_spec[iris_spec.size() - 1];
				cout << endl;
			}
			double sse1, sse2;
			//KM
			KMeans km;
			km.SetK(3);
			km.LoadSamples(iris_vector);
			km.Run(30);
			sse1 = km.CalculateSSE();

			//SAGMDE
			KMeans km2;
			KMeansSolution ksol;
			km2.SetK(3);
			km2.LoadSamples(iris_vector);
			km2.OptimalRun(ksol, 30);
			sse2 = ksol.GetSSE();
			cout << "Org Algo: SSE=" << sse1 << endl;
			cout << "Opt Algo: SSE=" << sse2 << endl;
		}

		void wine_cluster(int attrs=7,int cates=3,const char* path= "C:\\a\\seeds.txt") {
			vector<vector<double>> iris_parameters;
			vector<KVector> iris_vector;
			vector<int> iris_spec;

			//Load from file
			ifstream iris_dataset_file;
			iris_dataset_file.open(path);
			int cntw = 0;
			while (!iris_dataset_file.eof()) {
				cntw++;
				
				string file_line;
				stringstream csv_reader;
				iris_dataset_file >> file_line;
				for (int i = 0; i < file_line.size(); i++) {
					if (file_line[i] == ',')file_line[i] = ' ';
				}
				csv_reader << file_line;
				double temp;
				int temp2;
				iris_parameters.push_back(vector<double>{});
				iris_spec.push_back(0);
				KVector temp_vec;
				temp_vec.AllocateNew(attrs);
				temp_vec.Clear();
				iris_vector.push_back(temp_vec);
				//cout << "WINE:";
				for (int i = 0; i < attrs; i++) {
					csv_reader >> temp;
					iris_parameters[iris_parameters.size() - 1].push_back(temp);
					iris_vector[iris_vector.size() - 1].x[i] = temp;
					//cout << temp << ",";
				}
				csv_reader >> iris_spec[iris_spec.size() - 1];

				if (cntw % 50 == 0) {
					cout << "Readed lines:" << cntw << "," << file_line<< endl;
				}
				//cout << endl;
			}

			//Normalize the data
			for (int i = 0; i < attrs; i++) {
				double argmax = -1e+100, argmin = 1e+100;
				for (int j = 0; j < iris_parameters.size(); j++) {
					argmax = Max(argmax, iris_parameters[j][i]);
					argmin = Min(argmin, iris_parameters[j][i]);
				}
				for (int j = 0; j < iris_parameters.size(); j++) {
					iris_vector[j].x[i] = (iris_vector[j].x[i] - argmin) / (argmax - argmin);
				}
				cout << "ARGMAX:" << argmax << "," << "ARGMIN:" << argmin << endl;
			}
			/*
			for (int j = 0; j < iris_parameters.size(); j++) {
				cout << "WINE:";
				for (int i = 0; i < 15; i++) {
					cout << iris_vector[j].x[i] << ",";
				}
				cout << endl;

			}*/

			double sse1, sse2;
			//KM
			KMeans km;
			km.SetK(cates);
			km.LoadSamples(iris_vector);
			km.LoadSamplesTag(iris_spec);
			km.Run(30);
			sse1 = km.CalculateSSE();

			//SAGMDE
			KMeans km2;
			KMeansSolution ksol;
			km2.SetK(cates);
			km2.LoadSamples(iris_vector);
			km2.LoadSamplesTag(iris_spec);

			km2.OptimalRun(ksol, 30);
			sse2 = ksol.GetSSE();
			cout << "Org Algo: SSE=" << sse1 << endl;
			cout << "Opt Algo: SSE=" << sse2 << endl;
		}

		template<class T> T Min(const T& a, const T& b)const {
			return (a > b) ? b : a;
		}
		template<class T> T Max(const T& a, const T& b)const {
			return (a < b) ? b : a;
		}

		void show() {
			double sse1 = 0.0, sse2 = 0.0;
			vector<KVector> ptlist;
			Random R;
			int matSize = 575;
			Mat* cvMat = new Mat(matSize+1, matSize+1, CV_8UC3);
			Mat* cvMat2 = new Mat(matSize+1, matSize+1, CV_8UC3);
			for (int i = 0; i < matSize+1; i++) {
				for (int j = 0; j < matSize+1; j++) {
					cvMat->at<Vec3b>(static_cast<int>(i), static_cast<int>(j))[0] = 0;
					cvMat->at<Vec3b>(static_cast<int>(i), static_cast<int>(j))[1] = 0;
					cvMat->at<Vec3b>(static_cast<int>(i), static_cast<int>(j))[2] = 0;
				}
			}
			for (int t = 1; t <= 8; t++) {
				double center_x = R.UniformDistribution(0, static_cast<double>(matSize));
				double center_y = R.UniformDistribution(0, static_cast<double>(matSize));
				for (int i = 0; i < 50; i++) {
					Coord2d tmp = R.IrrNormalDistribution(center_x, 500, center_y, 500);
					KVector tmp2;
					tmp2.AllocateNew(2);
					tmp2.x[0] = tmp.x;
					tmp2.x[1] = tmp.y;
					if (tmp.x > 0.0 && tmp.x < static_cast<double>(matSize) && tmp.y>0 && tmp.y < static_cast<double>(matSize)) {
						ptlist.push_back(tmp2);
					}
					else {
						i--;
						continue;
					}
				}
			}
			//KM Algorithm

			km.SetK(8);
			km.LoadSamples(ptlist);
			km.Run(130);

			for (int i = 0; i < ptlist.size(); i++) {
				//cout << ptlist[i].x << "," << ptlist[i].y << ":" << km.cluster_result[i] << endl;
				cvMat->at<Vec3b>(static_cast<int>(ptlist[i].x[0]), static_cast<int>(ptlist[i].x[1]))[0] = color_preset[km.cluster_result[i]].x;
				cvMat->at<Vec3b>(static_cast<int>(ptlist[i].x[0]), static_cast<int>(ptlist[i].x[1]))[1] = color_preset[km.cluster_result[i]].y;
				cvMat->at<Vec3b>(static_cast<int>(ptlist[i].x[0]), static_cast<int>(ptlist[i].x[1]))[2] = color_preset[km.cluster_result[i]].z;

				cvMat->at<Vec3b>(static_cast<int>(ptlist[i].x[0] + 1), static_cast<int>(ptlist[i].x[1]))[0] = color_preset[km.cluster_result[i]].x;
				cvMat->at<Vec3b>(static_cast<int>(ptlist[i].x[0] + 1), static_cast<int>(ptlist[i].x[1]))[1] = color_preset[km.cluster_result[i]].y;
				cvMat->at<Vec3b>(static_cast<int>(ptlist[i].x[0] + 1), static_cast<int>(ptlist[i].x[1]))[2] = color_preset[km.cluster_result[i]].z;

				cvMat->at<Vec3b>(static_cast<int>(ptlist[i].x[0]), static_cast<int>(ptlist[i].x[1] + 1))[0] = color_preset[km.cluster_result[i]].x;
				cvMat->at<Vec3b>(static_cast<int>(ptlist[i].x[0]), static_cast<int>(ptlist[i].x[1] + 1))[1] = color_preset[km.cluster_result[i]].y;
				cvMat->at<Vec3b>(static_cast<int>(ptlist[i].x[0]), static_cast<int>(ptlist[i].x[1] + 1))[2] = color_preset[km.cluster_result[i]].z;

				cvMat->at<Vec3b>(static_cast<int>(ptlist[i].x[0] + 1), static_cast<int>(ptlist[i].x[1] + 1))[0] = color_preset[km.cluster_result[i]].x;
				cvMat->at<Vec3b>(static_cast<int>(ptlist[i].x[0] + 1), static_cast<int>(ptlist[i].x[1] + 1))[1] = color_preset[km.cluster_result[i]].y;
				cvMat->at<Vec3b>(static_cast<int>(ptlist[i].x[0] + 1), static_cast<int>(ptlist[i].x[1] + 1))[2] = color_preset[km.cluster_result[i]].z;
			}
			sse1 = km.CalculateSSE();
			imshow("KMeans Result", *cvMat);
			waitKey(0);

			//SAGMDE Algorithm
			for (int i = 0; i < matSize+1; i++) {
				for (int j = 0; j < matSize+1; j++) {
					cvMat2->at<Vec3b>(static_cast<int>(i), static_cast<int>(j))[0] = 0;
					cvMat2->at<Vec3b>(static_cast<int>(i), static_cast<int>(j))[1] = 0;
					cvMat2->at<Vec3b>(static_cast<int>(i), static_cast<int>(j))[2] = 0;
				}
			}
			
			kmOpt.SetK(8);
			kmOpt.LoadSamples(ptlist);
			KMeansSolution optimal_sol;
			kmOpt.OptimalRun(optimal_sol, 30, 41, 0.95, 1e+5, 1e+5, 1e+3, 1e+3,0.1,3);
			kmOpt.LoadFromSolution(optimal_sol);
			for (int i = 0; i < ptlist.size(); i++) {
				cvMat2->at<Vec3b>(static_cast<int>(ptlist[i].x[0]), static_cast<int>(ptlist[i].x[1]))[0] = color_preset[kmOpt.cluster_result[i]].x;
				cvMat2->at<Vec3b>(static_cast<int>(ptlist[i].x[0]), static_cast<int>(ptlist[i].x[1]))[1] = color_preset[kmOpt.cluster_result[i]].y;
				cvMat2->at<Vec3b>(static_cast<int>(ptlist[i].x[0]), static_cast<int>(ptlist[i].x[1]))[2] = color_preset[kmOpt.cluster_result[i]].z;

				cvMat2->at<Vec3b>(static_cast<int>(ptlist[i].x[0] +1), static_cast<int>(ptlist[i].x[1]))[0] = color_preset[kmOpt.cluster_result[i]].x;
				cvMat2->at<Vec3b>(static_cast<int>(ptlist[i].x[0] +1), static_cast<int>(ptlist[i].x[1]))[1] = color_preset[kmOpt.cluster_result[i]].y;
				cvMat2->at<Vec3b>(static_cast<int>(ptlist[i].x[0] +1), static_cast<int>(ptlist[i].x[1]))[2] = color_preset[kmOpt.cluster_result[i]].z;

				cvMat2->at<Vec3b>(static_cast<int>(ptlist[i].x[0]), static_cast<int>(ptlist[i].x[1] + 1))[0] = color_preset[kmOpt.cluster_result[i]].x;
				cvMat2->at<Vec3b>(static_cast<int>(ptlist[i].x[0]), static_cast<int>(ptlist[i].x[1] + 1))[1] = color_preset[kmOpt.cluster_result[i]].y;
				cvMat2->at<Vec3b>(static_cast<int>(ptlist[i].x[0]), static_cast<int>(ptlist[i].x[1] + 1))[2] = color_preset[kmOpt.cluster_result[i]].z;

				cvMat2->at<Vec3b>(static_cast<int>(ptlist[i].x[0] +1), static_cast<int>(ptlist[i].x[1] + 1))[0] = color_preset[kmOpt.cluster_result[i]].x;
				cvMat2->at<Vec3b>(static_cast<int>(ptlist[i].x[0] +1), static_cast<int>(ptlist[i].x[1] + 1))[1] = color_preset[kmOpt.cluster_result[i]].y;
				cvMat2->at<Vec3b>(static_cast<int>(ptlist[i].x[0] +1), static_cast<int>(ptlist[i].x[1] + 1))[2] = color_preset[kmOpt.cluster_result[i]].z;

			}
			sse2 = optimal_sol.GetSSE();
			cout << "Original SSE = " << sse1 << endl;
			cout << "Optimal SSE = " << sse2 << endl;

			vector<int> cluster_member_quantity;
			for (int i = 0; i < kmOpt.cluster_center.size(); i++) {
				cluster_member_quantity.push_back(0);
			}
			for (int i = 0; i < kmOpt.pointlist.size(); i++) {
				cluster_member_quantity[kmOpt.cluster_result[i]]++;
			}
			for (int i = 0; i < kmOpt.cluster_center.size(); i++) {
				cout << "In OptimalAlgo - CL " << i << ": Members=" << cluster_member_quantity[i] << endl;
			}
			
			imshow("KMeans+SA (SAGMDE) Result", *cvMat2);
			waitKey(0);
		}
	};
}

namespace ClusteringCG {
	//Image segmentation based on cluster algorithms
	class CGSegmentation {
	public:
		vector<Clustering::Coord3b> color_preset = { {255,100,100},{0,255,0},{0,0,255},{0,255,255},{255,0,255},{255,255,0},{255,255,255},{128,128,128} };

		Mat loaded_image;
		Mat* seg_image_km, *seg_image_sagmde;
		vector<Clustering::KVector> channels;
		void LoadImage(const char* file_path) {
			loaded_image = imread(file_path);
		}
		void VectorizeImage() {
			int h = loaded_image.rows;
			int w = loaded_image.cols;
			for (int i = 0; i < h; i++) {
				for (int j = 0; j < w; j++) {
					Vec3b vect;
					Clustering::KVector kvect;
					kvect.AllocateNew(3);
					vect = loaded_image.at<Vec3b>(i, j);
					kvect.Clear();
					kvect.x[0] = vect[0];
					kvect.x[1] = vect[1];
					kvect.x[2] = vect[2];
					channels.push_back(kvect);
				}
			}
		}
		double KMeansCluster(int cluster_channels = 3) {
			channels.clear();
			VectorizeImage();
			Clustering::KMeans km;
			km.SetK(cluster_channels);
			km.LoadSamples(channels);
			km.Run(30);
			seg_image_km = new Mat(loaded_image.rows, loaded_image.cols, CV_8UC3);
			int h = loaded_image.rows;
			int w = loaded_image.cols;
			int s = 0;
			for (int i = 0; i < h; i++) {
				for (int j = 0; j < w; j++) {
					seg_image_km->at<Vec3b>(i, j)[0] = color_preset[km.cluster_result[s]].x;
					seg_image_km->at<Vec3b>(i, j)[1] = color_preset[km.cluster_result[s]].y;
					seg_image_km->at<Vec3b>(i, j)[2] = color_preset[km.cluster_result[s]].z;

					s++;
				}
			}
			imshow("KMeans - CGSegmentation Result", *seg_image_km);
			waitKey(0);
			return km.CalculateSSE();
		}
		double SAGMDECluster(int cluster_channels = 3) {
			channels.clear();
			VectorizeImage();
			Clustering::KMeans km;
			Clustering::KMeansSolution kms;
			km.SetK(cluster_channels);
			km.LoadSamples(channels);
			km.OptimalRun(kms,30);

			seg_image_sagmde = new Mat(loaded_image.rows, loaded_image.cols, CV_8UC3);
			int h = loaded_image.rows;
			int w = loaded_image.cols;
			int s = 0;
			for (int i = 0; i < h; i++) {
				for (int j = 0; j < w; j++) {
					seg_image_sagmde->at<Vec3b>(i, j)[0] = color_preset[kms.cluster_result[s]].x;
					seg_image_sagmde->at<Vec3b>(i, j)[1] = color_preset[kms.cluster_result[s]].y;
					seg_image_sagmde->at<Vec3b>(i, j)[2] = color_preset[kms.cluster_result[s]].z;

					s++;
				}
			}
			imshow("SAGMDE - CGSegmentation Result", *seg_image_sagmde);
			waitKey(0);
			return kms.GetSSE();
		}

	};

}


int main()
{
	cout << setiosflags(ios::fixed) << setprecision(3);
	srand(time(NULL));
	Clustering::KMeansDemonstrator w;
	w.wine_cluster(15,5,"C:\\a\\artist.csv");
	return 0;
	cout << setiosflags(ios::fixed) << setprecision(3);
	srand(time(NULL));
	ClusteringCG::CGSegmentation cg;
	cg.LoadImage("C:\\a\\tj_cg_cluster2.jpg");
	double a = cg.KMeansCluster(4);
	double b = cg.SAGMDECluster(4);
	cout << "KMeans Error = " << a << endl;
	cout << "SAGMDE Error = " << b << endl;


}


