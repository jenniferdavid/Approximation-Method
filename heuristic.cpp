#include <fstream>
#include <sstream>
#include <math.h>
#include <iomanip> // needed for setw(int)
#include <string>
#include "stdio.h"
#include <iostream>
#include <cstdlib>
#include <Eigen/Dense>
#include <Eigen/LU> 
#include <vector>
#include <cmath>
#include <algorithm>
#include <time.h>
#include <limits>
#include "gnuplot_iostream.h"
#include <cstring>
#include <sys/stat.h>
#include <ctime>
#include <sys/types.h>
#include <locale.h>
#include <wchar.h>
#include <unistd.h>
#include <csignal>
#include <map>
#include "Hungarian.h"

using namespace std;
using namespace Eigen;

std::string outputpath = "/home/jendav/Videos/potts_spin_nn/Matrices/inputs/OUTPUTS/";
std::string inputpath = "/home/jendav/Videos/potts_spin_nn/Matrices/inputs/";
int nVehicles;
int nTasks;
int nDim;
int rDim;
Eigen::MatrixXd DeltaMatrix;
Eigen::MatrixXd TaskMatrix;
Eigen::MatrixXd EndMatrix;
Eigen::MatrixXd StartMatrix;
Eigen::MatrixXd EndHungMatrix;
Eigen::MatrixXd StartHungMatrix;
Eigen::VectorXd TVec; 
Eigen::VectorXd Edge; 
Eigen::VectorXd Edge2; 
Eigen::VectorXd Edge3; 
std::map<int,char> taskmap;

std::string task_alpha = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ`^&*()_+-=[]',./{}|:<>?;~!@#$abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ`^&*()_+-=[]',./{}|:<>?;~!@#$abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ`^&*()_+-=[]',./{}|:<>?;~!@#$abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ`^&*()_+-=[]',./{}|:<>?;~!@#$abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ`^&*()_+-=[]',./{}|:<>?;~!@#$abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ`^&*()_+-=[]',./{}|:<>?;~!@#$abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ`^&*()_+-=[]',./{}|:<>?;~!@#$abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ`^&*()_+-=[]',./{}|:<>?;~!@#$abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ`^&*()_+-=[]',./{}|:<>?;~!@#$abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ`^&*()_+-=[]',./{}|:<>?;~!@#$abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ`^&*()_+-=[]',./{}|:<>?;~!@#$abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ`^&*()_+-=[]',./{}|:<>?;~!@#$abcdef";


void DFS(Eigen::MatrixXd &TaskMatrix, int &vertex, vector<bool> &visited)
    {
        int noOfvisit = 1;
        visited[vertex] = true;
        noOfvisit++;
        Edge = VectorXd(nTasks-noOfvisit);
        cout << vertex+1 << " ";
        for(int i = 0; i < nTasks-noOfvisit; i++)
        {
            Edge(i) = TaskMatrix(vertex,i); //min dist from start to each of neighbour
        }
        
        MatrixXf::Index min;
        double minInd = Edge.minCoeff(&min);
        
            
//             int edge = TaskMatrix(vertex,i);
//             int neighbour = i;
//             if(edge == 10000000000){continue;}
//             if(visited[neighbour] == false)
//             {DFS(TaskMatrix, neighbour, visited);}
 //       }
    }
    
int main(int argc, const char* argv[])
    {
    
    clock_t tStart = clock();
    nVehicles = atoi(argv[1]);
    nTasks = atoi(argv[2]);    
    nDim = 2*nVehicles + nTasks;; 
    rDim = nVehicles + nTasks;
    
    time_t rawtime;
    struct tm * timeinfo;
    char buffer[40];
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    strftime(buffer, sizeof(buffer), "%d_%m_%Y_%I_%M_%S", timeinfo);
        
    std::string data = "M=" + std::to_string(nVehicles) +"_N="+std::to_string(nTasks);
    outputpath.append(std::string(data));
    std::string at = " AT TIME ";
    outputpath.append(std::string(at));
    outputpath.append(std::string(buffer));
    mkdir(outputpath.c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH );

    DeltaMatrix = MatrixXd::Ones(nDim,nDim);
    TaskMatrix = MatrixXd::Ones(nTasks,nTasks);
    TVec = VectorXd(nDim);
           
    std::string folder = inputpath + data;
    std::string folder1, folder2;

    if (!strcmp(argv[3],"-random"))
    {    
        for (int i=0; i<nDim; i++)
           {
               TVec(i) = 1;
           }
        cout << "\n TVec is: \n" << TVec <<endl;
    
        std::srand((unsigned int) time(0));
        DeltaMatrix = MatrixXd::Random(nDim,nDim);
        double HI = 100; // set HI and LO according to your problem.
        double LO = 1;
        double range= HI-LO;
        DeltaMatrix = (DeltaMatrix + MatrixXd::Constant(nDim,nDim,1.)) * range/2;
        DeltaMatrix = (DeltaMatrix + MatrixXd::Constant(nDim,nDim,LO));
        cout << "\n DeltaMatrix is: \n" << DeltaMatrix <<endl;
    }
         
    else if (!strcmp(argv[3],"-read"))
    {
        ifstream file;
        folder1 = folder+ "/tVec.txt"; 
        cout <<"\n"<<folder1<<endl;
        file.open(folder1); 
        if (file.is_open())
        {
           for (int i=0; i<nDim; i++)
           {
               double item;
               file >> item;
               TVec(i) = item;
           }
        }
        else
        {cout <<"\n tVec file not open"<<endl;
         return(0);
        }
        cout << "\n TVec is: \n" << TVec <<endl;
        ifstream file2;
        folder2 = folder+ "/deltaMat.txt"; 
        cout <<"\n"<<folder2<<endl;
        file2.open(folder2); 
        if (file2.is_open())
        {
            for (int i = 0; i < nDim; i++)
                for (int j = 0; j < nDim; j++)
                    {
                        double item2;
                        file2 >> item2;
                        DeltaMatrix(i,j) = item2;
                    }
        } 
        else
        {cout <<"\n Deltamat file not open"<<endl;
        return(0);
        }
        cout << "\n DeltaMatrix is: \n" << DeltaMatrix <<endl;
    }
    
    else 
    {cout << "\n Invalid option: " << argv[7] << "      exiting....\n";
                return(0);
    }
    
   
//     for (int i = 0; i < nDim; i++)
//     {    for (int j = 0; j < nDim; j++)
//             {
//                 if (DeltaMatrix(i,j) > DeltaMatrix(j,i))
//                     DeltaMatrix(i,j) = 10000000000; //removing multigraph 
//             }
//     }
    
    DeltaMatrix.diagonal().array() = 10000000000;
    DeltaMatrix.leftCols(nVehicles) *= 10000000000;
    DeltaMatrix.bottomRows(nVehicles) *= 10000000000;
    DeltaMatrix.topRightCorner(nVehicles,nVehicles) = DeltaMatrix.bottomLeftCorner(nVehicles,nVehicles).eval();       
    //DeltaMatrix.row(1) += 100* DeltaMatrix.row(0);
    //DeltaMatrix.triangularView<Lower>() *= 10000000000; //for predefined order
    //DeltaMatrix.topRows(nVehicles) = 0; //for vehicles 
    //DeltaMatrix.rightCols(nVehicles) = 0; //that have same initial cost

    ofstream outfile1;
    std::string createFile1 = "";    
    createFile1 = outputpath + "/" + "tVec" + ".txt";          
    outfile1.open(createFile1.c_str());     
    outfile1 << TVec << std::endl;
    outfile1.close();
    
    ofstream outfile2;
    std::string createFile2 = "";    
    createFile2 = outputpath + "/" + "deltaMat" + ".txt";          
    outfile2.open(createFile2.c_str());     
    outfile2 << DeltaMatrix << std::endl;
    outfile2.close();
    
    cout << "\n Updated DeltaMatrix/Adjacency Matrix is: \n" << DeltaMatrix << endl;    
    cout << "\n ///////////////////////////////////////////////////////////////////// " << endl;
    
   
    Eigen::MatrixXd TaskMatrix = DeltaMatrix.block(nVehicles,nVehicles,nTasks,nTasks);
    std::cout <<"TaskMatrix is\n" << TaskMatrix <<endl;
    
    int node = 0; // every task loop
    int start_node; //start node of every task loop
    std::vector<std::vector<int>> paths_list; //paths of each of the task
    std::vector <double> values; //minimum values of all the paths
    std::vector<std::vector<double>> paths_values_list; //paths of each of the task

    LABEL0:
    start_node = node;
    std::vector<int> visited; //Global unvisited vertices vector
    std::vector <int> paths;
    paths.push_back(node);
    std::vector <double>value_paths; //stores the values of each of the path of each task
    double sum_of_elems = 0;
    int visitNo = 0;
    LABEL: 
    visitNo++;
    visited.push_back(start_node);
    Edge = VectorXd((nTasks)); 
    for(int i = 0; i < ((nTasks)); i++)
        {
            if (start_node==i)
                Edge(i) = INT_MAX;
            else if (std::find(visited.begin(),visited.end(),i) != visited.end() )
                Edge(i) = INT_MAX;
            else
            {Edge(i) = TaskMatrix(start_node,i);} //min dist from start to each of neighbour
    cout<<"\nEdge is"<<Edge(i)<<endl;
        } 
    
    MatrixXf::Index minI;
    double minValue = Edge.minCoeff(&minI);
    cout<<"\nminVal is"<<minValue<<endl;
    cout<<"\nminI is"<<minI<<endl;
    paths.push_back(minI);
    value_paths.push_back(minValue);
    cout<<"\nvisited vertices are"<<visitNo<<endl;    
    start_node = minI;
    //value of the path    
    
    if (visitNo == (nTasks-1))
        {node++;
        paths_list.push_back(paths);
        paths_values_list.push_back(value_paths);
        std::for_each(value_paths.begin(), value_paths.end(), [&] (double n) 
        {sum_of_elems += n;});
        values.push_back(sum_of_elems);
        cout<<"\n..................."<<endl;    
        if (node == (nTasks))
            {goto LABEL1;} //paths for all the tasks found
        else
            goto LABEL0; //run for the next task available
        }
    else
        {goto LABEL;} //run till all the tasks are done for that task loop
    LABEL1:

    for ( std::vector<std::vector<int>>::size_type i = 0; i < paths_list.size(); i++ )
    {for ( std::vector<int>::size_type j = 0; j < paths_list[i].size(); j++ )
    {std::cout << paths_list[i][j] << ' ';}
    std::cout << std::endl;}
            
    for ( std::vector<std::vector<int>>::size_type i = 0; i < paths_values_list.size(); i++ )
    {for ( std::vector<int>::size_type j = 0; j < paths_values_list[i].size(); j++ )
    {std::cout << paths_values_list[i][j] << ' ';}
    std::cout << std::endl;}

    std::cout << "The smallest path is " << *std::min_element(std::begin(values),std::end(values)) << '\n';
    auto smallest = std::min_element(values.begin(), values.end() );
    
    double avg1 = std::accumulate(values.begin(), values.end(), 0LL) / values.size();
    std::cout << "The avg is: " << avg1 << endl;
    
    std::vector<double>::iterator first = values.begin();
    int  path_index = std::distance(first,smallest);
    std::cout << "The distance is: " << path_index << endl;
  
    for ( std::vector<int>::size_type j = 0; j < nTasks; j++ )
    {std::cout << paths_list[path_index][j] << ' ';}
    std::cout << std::endl; // the path array
             
    for ( std::vector<int>::size_type j = 0; j < paths_values_list[path_index].size(); j++ )
    {   
        std::cout << paths_values_list[path_index][j] << ' ';
    } //the path array with values

    std::vector<std::pair <int,int>> edge;
    
    double chunk_length = *smallest/(nVehicles);
    cout << "\nchunk_length " << chunk_length<< endl;
    cout <<"............................." <<endl;
    double val = 0;
    double addChunk = 0;
    int count =0;
    vector<int>cut;
   // cut.push_back(0);
    
    for ( std::vector<int>::size_type j = 0; j < paths_values_list[path_index].size(); j++ )
    {  
       if (count != (nVehicles-1))
       {
       cout << "\nj is" << j<<endl;
       val = val + paths_values_list[path_index][j];
       addChunk = paths_values_list[path_index][j];
       cout << "val" << val<< endl;
       cout << "addChunk is" << addChunk<< endl;
       
//        if ((addChunk > chunk_length) && (val >chunk_length))
//        {
//            cout << "\naddChunk is big and chunk length reached"<<endl;
//            if (j!=0)
//            {cut.push_back(j-1);
//                cout <<"pushing j-1"<<endl;}
//            else {cut.push_back(j);
//                cout <<"pushing j"<<endl;}
//            count ++;
//            val = 0;
//            addChunk = 0;
//        }
       
       if (addChunk > chunk_length)
       {
           
           cout << "\naddChunk is big"<<endl;
           cout << "adding j= "<<j-1<<endl;
           cut.push_back(j-1);
           /*if (val > chunk_length)
           {cut.push_back(j);
            cout << "adding j= "<<j<<endl;}
           */count ++;
           val = 0;
           
       }
       
       if (val > chunk_length)
       {
        cout << "\nmin chunk length reached"<<endl;
        cut.push_back(j-1);
        cout << "adding j-1= "<<j-1<<endl;
        count++;
        val = 0;
        if ((j != 0) || (j != paths_values_list[path_index].size()-1))
        j=j-1;
        else
            break;
        }
       }
    } 
    int check = 0;
    cout <<"before"<<endl;
    for (int i=0; i<cut.size(); i++) 
    { 
        cout <<cut[i] << " " ;
    } 
    if ( std::any_of(cut.begin(), cut.end(), [](int i){return i<0;}))
    {check=1;}
    
    cut.push_back(nTasks-1);
    cout << "\nafter" <<endl;
    for (int i=0; i<cut.size(); i++) 
    { 
       if (check ==1)
        { cut[i] = cut[i]+1;}
         cout <<cut[i] << " ";
    } 
    int cutNo = cut.size();
    cout <<"\n no of cuts:"<<cutNo << endl;
   
    
    vector<double> temp_path;
    vector<double> temp_path_value;
    std::vector <std::vector<double>> chunks; //should be equal to number of vehicles
    std::vector <std::vector<double>> chunks_path; //should be equal to number of vehicles
    
    int i =0;
    for ( std::vector<int>::size_type j = 0; j < paths_list.size(); j++ )
        {  temp_path.push_back(paths_list[path_index][j]);
           if (j == cut[i])
           { chunks.push_back(temp_path); i++;
               temp_path.clear();}
        }
        
//     for ( std::vector<std::vector<int>>::size_type i = 0; i < temp_path.size(); i++ )
//     {std::cout << temp_path[i] << ' ';
//     std::cout << std::endl;}
    
    for ( std::vector<std::vector<int>>::size_type i = 0; i < chunks.size(); i++ )
    {for ( std::vector<int>::size_type j = 0; j < chunks[i].size(); j++ )
    {std::cout << chunks[i][j] << ' ';}
    std::cout << std::endl;}
        cout << "\n////////////////////////////////////////////////////////////////" << endl;

    //start points
    vector<int> start_points;
    vector<int> end_points;
    
    for ( std::vector<std::vector<int>>::size_type i = 0; i < chunks.size(); i++ )
    {start_points.push_back(chunks[i][0]);
        cout << start_points[i]<<endl;
    end_points.push_back(chunks[i][(chunks[i].size())-1]);
        cout << end_points[i]<<endl;
    }
    
        cout << "\n////////////////////////////////////////////////////////////////" << endl;

    std::cout <<"DeltaMatrix is\n" << DeltaMatrix <<endl;

    Eigen::MatrixXd StartMatrix = DeltaMatrix.block(0,nVehicles,nVehicles,nTasks);
    std::cout <<"StartMatrix is\n" << StartMatrix <<endl;
    
    Eigen::MatrixXd EndMatrix = DeltaMatrix.block(nVehicles,nVehicles+nTasks,nTasks,nVehicles);
    std::cout <<"EndMatrix is\n" << EndMatrix <<endl;
    
    Eigen::MatrixXd StartHungMatrix = MatrixXd::Zero(nVehicles,nVehicles);
    Eigen::MatrixXd EndHungMatrix = MatrixXd::Zero(nVehicles,nVehicles);
    
    std::vector<std::vector<double>> v1;
    std::vector<std::vector<double>> v2;
    std::vector<double> v3;
    std::vector<double> v4;
    
    for (int i = 0; i < cutNo; i++)
        {
            for (int j = 0; j < cutNo; j++)
            {
                 StartHungMatrix(i,j) =  StartMatrix(j,start_points[i]);  
                 v3.push_back(StartMatrix(j,start_points[i]));
            }
            v1.push_back(v3);
            v3.clear();
        }
    std::cout <<"StartHungMatrix is\n" << StartHungMatrix <<endl;
    
    for (int i = 0; i < cutNo; i++)
        {
            for (int j = 0; j < cutNo; j++)
            {
                 EndHungMatrix(i,j) = EndMatrix(end_points[i],j); 
                 v4.push_back(EndMatrix(end_points[i],j));
            }
            v2.push_back(v4);
            v4.clear();
        }
    std::cout <<"EndHungMatrix is\n" << EndHungMatrix <<endl;
    cout << "\n////////////////////////////////////////////////////////////////" << endl;
    
    for ( std::vector<std::vector<int>>::size_type i = 0; i < v1.size(); i++ )
    {for ( std::vector<int>::size_type j = 0; j < v1[i].size(); j++ )
    {std::cout << v1[i][j] << ' ';}
    std::cout << std::endl;}
        
    cout << "\n////////////////////////////////////////////////////////////////" << endl;

    for ( std::vector<std::vector<int>>::size_type i = 0; i < v2.size(); i++ )
    {for ( std::vector<int>::size_type j = 0; j < v2[i].size(); j++ )
    {std::cout << v2[i][j] << ' ';}
    std::cout << std::endl;}    

    HungarianAlgorithm HungAlgo;
    vector<int> assignment;
    vector<int> assignment2;
    double start_cost = HungAlgo.Solve(v1, assignment);
    double end_cost = HungAlgo.Solve(v2, assignment2);
    
    for (unsigned int x = 0; x < v1.size(); x++)
    {std::cout << x << "," << assignment[x] << "\t";}
    
        cout << "\n////////////////////////////////////////////////////////////////" << endl;

    for (unsigned int x = 0; x < v2.size(); x++)
    {std::cout << x << "," << assignment2[x] << "\t";}
    


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

    printf("\nTotal computational time taken: %.2f\n", (((double)(clock() - tStart)/CLOCKS_PER_SEC)));
    
    }
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
// 
//         
//         
//     {  
//         temp_path.push_back(paths_list[path_index][j]);
//      cout << temp_path[j]<< " ";
//     }
//     chunks.push_back(temp_path);
//     
                   

//cut has the cuts to be made on paths_list
//for ex - 4veh-10task
//457168302 cut by 234 becomes
//457 x 1 x 6 x 8302
//vector of vectors is created;
//each chunk is taken - starting vector and end vector
//all starting vector - with StartMatrix compared for minimum
//all end vector - with EndMatrix compared for minimum
//

   // }
    
/*
 * 
    for ( std::vector<int>::size_type j = 0; j < nTasks-1; j++ )
    {edge.push_back(std::make_pair(paths_list[path_index][j],paths_list[path_index][j+1]));}
    for (int i=0; i<nTasks-1; i++) 
    {cout <<"\n"<< edge[i].first << " " << edge[i].second << endl;}
//    
//       std::cout << "here " << endl;
// 
//     if (nVehicles > nTasks)
//     {   
//         nVehicles = nTasks;
//         for ( std::vector<int>::size_type j = 0; j < paths_values_list[path_index].size(); j++ )
//         {    
//         vector <double> temp;
//         temp.push_back(paths_values_list[path_index][j]);
//         chunks.push_back(temp);
//         }
//     }
//     
// //  for (std::vector<double>::iterator it = chunks.begin() ; it != chunks.end(); ++it)
// //     std::cout << ' ' << *it;
//     
//     else if (nVehicles == 1)
//     {  
//         vector <double> temp;
//         // in this case there are no chunks  
//         for ( std::vector<int>::size_type j = 0; j < paths_values_list[path_index].size(); j++ )
//         {    
//         temp.push_back(paths_values_list[path_index][j]);
//         }
//         chunks.push_back(temp);
//     }
//     
//     else if (nVehicles == (nTasks-1))
//     {
//         for ( std::vector<int>::size_type j = 0; j < paths_values_list[path_index].size(); j++ )
//         {prechunks.push_back(paths_values_list[path_index][j]);}
//         std::cout << "\nThe biggest task cost is " << *std::max_element(std::begin(prechunks),std::end(prechunks)) << '\n';
//         auto biggest = std::max_element(prechunks.begin(), prechunks.end());
//         std::vector<double>::iterator first = prechunks.begin();
//         int divisor = std::distance(first,biggest);
//         std::cout << "The cut point is: " << divisor << '\n';   
//         vector <double> temp;
//         temp.push_back(edge[divisor].first);
//         temp.push_back(edge[divisor].second);        
//         chunks.push_back(temp);
//             //element found and so remove it tasks
//     }
//     
//     for ( std::vector<int>::size_type j = 0; j < paths_values_list[path_index].size(); j++ )
//         {prechunks.push_back(paths_values_list[path_index][j]);}
//         std::cout << "\nThe biggest task cost is " << *std::max_element(std::begin(prechunks),std::end(prechunks)) << '\n';
//         auto biggest = std::max_element(prechunks.begin(), prechunks.end());
//         std::vector<double>::iterator first2 = prechunks.begin();
//         int divisor2 = std::distance(first2,biggest);
//         std::cout << "The cut point for big task is: " << divisor2 << '\n'; 
// 
//     for ( std::vector<std::vector<int>>::size_type i = 0; i < paths.size(); i++ )
//     {std::cout << paths[i] << ' ';
//     std::cout << std::endl;}
//             cout <<"here"<<endl;
//             
//              cout <<"here"<<endl;
//     for ( std::vector<std::vector<int>>::size_type i = 0; i < value_paths.size(); i++ )
//     {std::cout << value_paths[i] << ' ';
//     std::cout << std::endl;}
//             cout <<"here"<<endl;
//     
//     for ( std::vector<std::vector<int>>::size_type i = 0; i < values.size(); i++ )
//     {std::cout << values[i] << ' ';
//     std::cout << std::endl;}
  int a=1;
  for(auto it = task_alpha.begin(); it != task_alpha.end(); ++it) 
  {
      taskmap[a] = *it;
      a++;
      while (!taskmap.empty())
        {
        //  std::cout << taskmap.begin()->first << " => " << taskmap.begin()->second << '\n';
            taskmap.erase(taskmap.begin());
        }
  }

//   for( std::vector<int>::size_type j = 0; j < nTasks; j++ )
//   {
//      int b = paths_list[path_index][j];
//      cout << taskmap.begin()->first << endl;
//   }
          cout <<"here1"<<endl;

   for (std::map<int,char>::iterator it=taskmap.begin(); it!=taskmap.end(); ++it)
        {cout <<"here2"<<endl;
         std::cout << it->first << " => " << it->second << '\n';}

         std::cout << taskmap.begin()->first << " => " << taskmap.begin()->second << '\n';
 * for ( std::vector<int>::size_type j = 0; j < nTasks; j++ )
    {
       int b = paths_list[path_index][j];
       cout <<taskmap.find(b)->first << endl;
    }
 for ( std::vector<std::vector<int>>::size_type i = path_index; i < path_index+nTasks; i++ )
    {std::cout << paths[i] << ' ';
    std::cout << std::endl;}
            cout <<"here"<<endl;*/

// //     for(int start = 0; start <nTasks; start++)
// //    {
//         cout <<"\n.............."<<endl;
//         // Create a "visited" array (true or false) to keep track of if we visited a vertex.
//         int start = 1;
//         vector<bool> visited;
//         vector<int> path; //visited vertices
//         for(int i = 0; i < nTasks; i++)
//         {visited.push_back(false);}
//         path.push_back(start);
//        // unvisited.pop_back(start);
//         //start the graph
//         int noOfvisit = 0;
//         visited[start] = true;
//         noOfvisit++;
//         Edge = VectorXd(nTasks);
//         for(int i = 0; i < (nTasks); i++)
//         {
//             if (start==i)
//                 Edge(i) = INT_MAX;
//             else
//             {Edge(i) = TaskMatrix(start,i); //min dist from start to each of neighbour
//             cout<<"\nEdge is"<<Edge(i)<<endl;}
//         }        
//         MatrixXf::Index minI;
//         double minValue = Edge.minCoeff(&minI);
//         cout<<"\nminVal is"<<minValue<<endl;
//         cout<<"\nminI is"<<minI<<endl;
//         
//         int new_vertex = minI+1;
//         noOfvisit++;
//         path.push_back(minI);
//         Edge2 = VectorXd(nTasks);
//         
//         for(int i = 0; i < (nTasks); i++)
//         {
//             if (new_vertex==i || i<new_vertex)
//                 Edge2(i) = INT_MAX;
//             else
//             {Edge2(i) = TaskMatrix(new_vertex,i); //min dist from start to each of neighbour
//             cout<<"\nEdge is"<<Edge2(i)<<endl;}
//         }        
//         MatrixXf::Index minI2;
//         double minValue2 = Edge2.minCoeff(&minI2);
//         cout<<"\nminVal2 is"<<minValue2<<endl;
//         cout<<"\nminI2 is"<<minI2<<endl;
//         
//         int new_vertex2 =minI2+1;
//         noOfvisit++;
//         path.push_back(minI2);
//         Edge3 = VectorXd(nTasks); // should be size of no of unvisited vertices
//         
//         for(int i = 0; i < (nTasks); i++)
//         {
//             if (new_vertex2==i || i<new_vertex2)
//                 Edge3(i) = INT_MAX;
//             else                               
//             {Edge3(i ) = TaskMatrix(new_vertex2,i); //min dist from start to each of neighbour
//             cout<<"\nEdge is"<<Edge3(i)<<endl;}
//         }        
//         MatrixXf::Index minI3;
//         double minValue3 = Edge3.minCoeff(&minI3);
//         cout<<"\nminVal3 is"<<minValue3<<endl;
//         cout<<"\nminI3 is"<<minI3<<endl;
//         
//                     cout<<"\n/////////////////"<<endl;

//             int edge = TaskMatrix(vertex,i);
//             int neighbour = i;
//             if(edge == 10000000000){continue;}
//             if(visited[neighbour] == false)
//             {DFS(TaskMatrix, neighbour, visited);}
        
        
//     }
//}
 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
