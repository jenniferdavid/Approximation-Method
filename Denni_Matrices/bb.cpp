#include <fstream>
#include <math.h>
#include <iomanip> // needed for setw(int)
#include <string>
#include "stdio.h"
#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <time.h>
#include <limits>

using namespace std;

int main(int argc, const char* argv[]){
    vector<pair<int,int>> index;
    vector<pair<int,int>>::iterator it;
    int row,col;
    float A[3][3], solution[3][3];
    A[0][0] = 0;
    A[0][1] = 0;
    A[0][2] = 0;
    A[1][0] = 0.5;
    A[1][1] = 0.5;
    A[1][2] = 0;
    A[2][0] = 0.5;
    A[2][1] = 0.5;
    A[2][2] = 1;
    
    
    
    for(int i=0;i<3;i++){
    for(int j=0;j<3;j++){
        solution[i][j] = 0;
    }
    }

    std::cout << "start !" << std::endl;
    
    for(int i=0;i<3;i++){
    for(int j=0;j<3;j++){
        std::cout << A[i][j] << std::endl;
    }
    }
    
    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){
            if(A[i][j] == 0.5)
            {
                std::cout << "pushing";
                index.push_back(pair<int,int>(i,j));
            }
        }
    }
    std::cout << "Index array is "<< std::endl;
    int kk = 0;
    for (it = index.begin(); it != index.end(); it++,kk++ ) {
        std::cout << index.at(kk).first<< "," << index.at(kk).second << std::endl;
    }
    
    int nSol = 0;
 while(!index.empty() && nSol != 2){
        vector<pair<int,int>> indexCopy = index;
        std::cout << "starting with " << index.back().first<< "," << index.back().second << " index size is " << index.size()<< std::endl;
        row = index.back().first;
        col = index.back().second;
        indexCopy.pop_back();
        index.pop_back();
        solution[row][col] = 1;
        std::cout << " index size is " << index.size() << std::endl;
        std::cout << "Index array is "<< std::endl;
        kk = 0;
        for (it = index.begin(); it != index.end(); it++,kk++ ) {
            std::cout << index.at(kk).first<< "," << index.at(kk).second << std::endl;
        }
        if(!index.empty()){
            int k = 0;
            for (it = indexCopy.begin(); it != indexCopy.end(); it++,k++) {
                std::cout << "checking" << indexCopy.at(k).first<< "," << indexCopy.at(k).second << std::endl;
                if(indexCopy.at(k).first == row || indexCopy.at(k).second == col){
                    std::cout << "deleting index " << k << " with data " << indexCopy.at(k).first << "," << indexCopy.at(k).second << std::endl;
                    indexCopy.erase(it);
                    it = indexCopy.begin();
                    k=0;
                }
            }
        }
        std::cout << "dddddd"<<std::endl;
        int a = 0;
        for (it = indexCopy.begin(); it != indexCopy.end(); it++,a++ ) {
            std::cout << "leftover" << indexCopy.at(a).first<< "," << indexCopy.at(a).second << std::endl;
            solution[indexCopy.at(a).first][indexCopy.at(a).second] = 1;
        }
        std::cout << "Found a solution !!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
        //printing array
        for(int i=0;i<3;i++){
            for(int j=0;j<3;j++){
                std::cout << solution[i][j] << ",";
                solution[i][j] = 0;
            }
            std::cout << std::endl;
        }
        nSol++;
    }
}