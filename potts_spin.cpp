/* Trials with Potts_Spin based optimization.
 *
 * Copyright (C) 2014 Jennifer David. All rights reserved.
 *
 * BSD license:
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of
 *    contributors to this software may be used to endorse or promote
 *    products derived from this software without specific prior written
 *    permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR THE CONTRIBUTORS TO THIS SOFTWARE BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
   \file potts_spin.cpp
   
   Trials with running neural network based optimization method for task assignment. 
   
*/
#include <fstream>
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
#include <fstream>
#include <algorithm>
#include <time.h>
#include <limits>
#include "gnuplot_iostream.h"

using namespace std;
using namespace Eigen;

int nVehicles;
int nTasks;
int nDim;
int rDim;
double kT_start;
double kT_stop;
double kT_fac;
double g;
double kT;
double lk;
float kappa;
double beta = 1;

bool PState;
bool LoopState;

double kT_swfac = 0.0015;
double kT_fac2;

int TaskState;
int UpdateState;

static const double small = 1e-15;
static const double onemsmall = 1 - small;
static const double lk0 = 1/small - 1; 

////////////////////////////////////////////////////////////////////////////////////////

Eigen::MatrixXd VMatrix;
Eigen::MatrixXd DeltaMatrix;
Eigen::MatrixXd PMatrix;
Eigen::MatrixXd I = MatrixXd::Identity(nDim,nDim); 
Eigen::MatrixXd P;
Eigen::MatrixXd NonNormUpdatedVMatrix;
Eigen::MatrixXd UpdatedVMatrix;
Eigen::MatrixXd UpdatedPMatrix;
Eigen::MatrixXd E_task; 
Eigen::MatrixXd E_loop; 
Eigen::MatrixXd E_local; 
Eigen::MatrixXd vDeltaR;
Eigen::MatrixXd vDeltaL;

Eigen::VectorXd TVec; 
Eigen::VectorXd vdVecL(nDim);
Eigen::VectorXd vdVecR(nDim);
Eigen::VectorXd rightVec;
Eigen::VectorXd leftVec;
Eigen::VectorXd sumv;
Eigen::VectorXd sumw; 
Eigen::VectorXd sumr; 
Eigen::VectorXd sumc;
Eigen::VectorXd cTime;
Eigen::VectorXd checkTimeVec;
Eigen::VectorXd sub;
Eigen::MatrixXd vMatBest; 
double costValueBest = 10000000000; 
////////////////////////////////////////////////////////////////////////////////////////////////
  
Eigen::MatrixXd normalisation(Eigen::MatrixXd & VMatrix) //this function normalises a matrix - makes into doubly stochastic matrix
  {
      int y = 0;
      LABEL0:
      sumr = VMatrix.rowwise().sum();
      for (int i = 0; i < nDim; i++)
                {
                    if (sumr(i) == 0.000)
                        { 
                            //cout << "\n Row " << k << " is with constraints, so skipping" << endl;
                        }
                    else if (sumr(i) == 1.000)
                        { 
                            //cout << "\n Row " << k << " is already normalised" << endl;
                        }
                    else 
                        {   // cout << "\n Row Normalising " << endl;
                            for (int j = 0; j < nDim; j++)
                            {VMatrix(i,j) = VMatrix(i,j)/sumr(i);}
                        }
                 }                  
       //cout << "\n So finally, the Row Normalised UpdatedVMatrix is \n" << UpdatedVMatrix << endl;
       //cout << "\n row sum after row normalisation is \n" << UpdatedVMatrix.rowwise().sum() << endl;
   
       //Normalising columns of VMatrix
       sumc = VMatrix.colwise().sum();
       for (int i = 0; i < nDim; i++)
                {
                    if (sumc(i) == 0)
                        {//cout << "\n Col " << k << " is with constraints, so skipping" << endl;
                        }
                    else if (sumc(i) == 1)
                        {//cout << "\n Col " << k << " is already normalised" << endl;
                        }   
                    else 
                        {//cout << "\n Column Normalising \n" << endl;
                            for (int j = 0; j < nDim; j++)
                                {VMatrix(j,i) = VMatrix(j,i)/sumc(i);}
                        }
                 }
       y++;
       if (y != 200)
            {goto LABEL0;}
       else
            {cout << "\n Normalised VMatrix is: \n" << VMatrix << endl;
            cout << "Sum of VMatrix after row normalisation \n" << VMatrix.rowwise().sum() << endl;
            cout << "Sum of VMatrix after col normalisation \n" << VMatrix.colwise().sum() << endl;}
       return VMatrix;
  }
  
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Eigen::MatrixXd getVMatrix (int &nVehicles, int &nTasks, int &nDim, int &rDim) //initialize VMatrix
  {
      cout << "\n ///////////////////////////////////////////////////////////////////// " << endl;
      cout << "\n No. of Vehicles M = " << nVehicles << endl;
      cout << "\n No. of Tasks N = " << nTasks << endl;
      cout << "\n Dimension of VMatrix is: " << nDim << endl;
      cout << "\n Total no of Vij: " << nDim*nDim << endl;
        
      VMatrix = MatrixXd::Zero(nDim,nDim);
      
      double tmp = 1./(rDim-1);
      cout << "\n tmp is: " << tmp << endl;

      // intitalising VMatrix with 1/n values
      for (int i = 0; i < nDim; i++)
        {
            for (int j = 0; j < nDim; j++)
            {
                VMatrix(i,j) = tmp + (((double) rand() / (RAND_MAX))/80) ;// + 0.02*(rand() - 0.5);	// +-1 % noise
            }
        }
    
      //adding all the constraints for the vehicles and tasks
      VMatrix.diagonal().array() = 0;
      VMatrix.leftCols(nVehicles) *= 0;
      VMatrix.bottomRows(nVehicles) *= 0;
      VMatrix.topRightCorner(nVehicles,nVehicles) *= 0;  
      
      VMatrix = normalisation(VMatrix);
      
//       VMatrix << 
// 0,0,1,0,0,0,0,0,0,0,
// 0,0,0,1,0,0,0,0,0,0,
// 0,0,0,0,0,0,1,0,0,0,
// 0,0,0,0,0,0,0,0,1,0,
// 0,0,0,0,0,0,0,1,0,0,
// 0,0,0,0,1,0,0,0,0,0,
// 0,0,0,0,0,1,0,0,0,0,
// 0,0,0,0,0,0,0,0,0,1,
// 0,0,0,0,0,0,0,0,0,0,
// 0,0,0,0,0,0,0,0,0,0;


      return VMatrix;
  }
  
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
Eigen::MatrixXd asyncRUpdate(Eigen::MatrixXd & E_local, Eigen::MatrixXd & UpdatedVMatrix)
{
            Eigen::MatrixXd E = MatrixXd::Zero(nDim,nDim);
            //E = ((-1*E_local)/kT);
            int ran = rand() % 2;
            cout << "\n ran is: " << ran << endl;         

            if (ran == 1)
            {
                int RowUp = rDim;
                int RowLo = 0;
                int RowRange = abs(RowUp - RowLo);
                int RowRan = rand()% (RowRange) + RowLo;
                cout << "\n RowRan is: " << RowRan << endl;         
                
                cout << "\n Updating Row" << endl;
                for (int j = nVehicles; j < nDim; j++)
                    {   E(RowRan,j) = ((-1*(E_local(RowRan,j) - (beta*(UpdatedVMatrix(RowRan,j)))))/kT);
                        UpdatedVMatrix(RowRan,j) = std::exp (E(RowRan,j));}  
                cout << "\n Updating Row VMatrix is: (the numerator) \n" << UpdatedVMatrix << endl;
                    
                Eigen::VectorXd sumv = UpdatedVMatrix.rowwise().sum();
                sumv = UpdatedVMatrix.rowwise().sum();
                cout << "\n sumv is (my ref): \n" << sumv << endl;
            
                for (int j = nVehicles; j < nDim; j++)
                    {UpdatedVMatrix(RowRan,j) = UpdatedVMatrix(RowRan,j)/sumv(RowRan);}  
                cout << "\n Updating Row VMatrix is (complete update):  \n" << UpdatedVMatrix << endl;
                
            }
            
            else
            {    
                int ColUp = nDim;
                int ColLo = nVehicles;
                int ColRange = abs(ColUp - ColLo);
                int ColRan = rand()% (ColRange) + ColLo;
                cout << "\n ColRan is: " << ColRan << endl;         
                             
                cout << "\n Updating Col" << endl;
                for (int j = 0; j < rDim; j++)
                    {   E(j,ColRan) = ((-1*(E_local(j,ColRan) - (beta*(UpdatedVMatrix(j,ColRan)))))/kT);
                        UpdatedVMatrix(j,ColRan) = std::exp (E(j,ColRan));}  
                cout << "\n Updating Column VMatrix is: (the numerator) \n" << UpdatedVMatrix << endl;

                Eigen::VectorXd sumw = UpdatedVMatrix.rowwise().sum();
                sumw = UpdatedVMatrix.colwise().sum();
                cout << "\n sumw is (my ref): \n" << sumw << endl;
            
                for (int j = 0; j < rDim; j++)
                    {UpdatedVMatrix(j,ColRan) = UpdatedVMatrix(j,ColRan)/sumw(ColRan);}  
                cout << "\n Updating Column VMatrix is (complete update):  \n" << UpdatedVMatrix << endl;
                
            }
                return UpdatedVMatrix;

}
    
Eigen::MatrixXd syncUpdate(Eigen::MatrixXd & E_local, Eigen::MatrixXd & UpdatedVMatrix) //Updating mean field equations (VMatrix) along row-wise
  {
        Eigen::MatrixXd E = MatrixXd::Zero(nDim,nDim);
        // E = ((-1*E_local)/kT);

        for (int i = 0; i < rDim; i++)
             {
               for (int j=nVehicles; j< nDim; j++)
                    {
                        E(i,j) = ((-1*(E_local(i,j) - (beta*(UpdatedVMatrix(i,j)))))/kT);
                        UpdatedVMatrix(i,j) = std::exp (E(i,j));
                    }  
             }
        cout << "\n Updating VMatrix is: (the numerator) \n" << UpdatedVMatrix << endl;
        
        Eigen::VectorXd sumv = UpdatedVMatrix.rowwise().sum();
        cout << "sumv is (my ref): \n " << sumv << endl;
        for (int i = 0; i < rDim; i++)
             {                    
               for (int j=nVehicles; j< nDim; j++)
                    {
                       UpdatedVMatrix(i,j) /= sumv(i);
                    }  
             }          
        UpdatedVMatrix.topRightCorner(nVehicles,nVehicles) *= 0;  
        cout << "\n Updating Row VMatrix is (complete update):  \n" << UpdatedVMatrix << endl;
       
        return UpdatedVMatrix;
    }

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::tuple<Eigen::VectorXd, Eigen::VectorXd> calculateLR(Eigen::MatrixXd & VMatrix, Eigen::MatrixXd & PMatrix) //calculate LR from V and PMatrix
   {
        vDeltaL = VMatrix.transpose() * DeltaMatrix;
        cout << "\n vDeltaL is: \n" << vDeltaL << endl;
        vdVecL = vDeltaL.diagonal();
        cout << "\n vdVecL is: \n" << vdVecL << endl;
        leftVec = PMatrix.transpose() * (TVec + vdVecL);
        cout << "\n leftVec is: \n" << leftVec << endl;

        vDeltaR = VMatrix * DeltaMatrix.transpose();
        cout << "\n vDeltaR is: \n" << vDeltaR << endl;
        vdVecR = vDeltaR.diagonal();
        cout << "\n vdVecR is: \n" << vdVecR << endl;
        rightVec = PMatrix * (TVec +vdVecR);
        cout << "\n rightVec is: \n" << rightVec << endl;

        return std::make_tuple(leftVec, rightVec);
   }

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
Eigen::MatrixXd calculateE(Eigen::VectorXd &leftVec, Eigen::VectorXd & rightVec, Eigen::MatrixXd & PMatrix, int & TaskState, bool & LoopState) //calculate ELocal using L,R and PMatrix
    {
        
     if (TaskState == 1)
     {
        MatrixXf::Index maxl, maxr;
        double maxleftVecInd, maxrightVecInd;
        maxleftVecInd = leftVec.maxCoeff(&maxl);
        maxrightVecInd = rightVec.maxCoeff(&maxr);
        cout << "\n maxr is: " << maxr << endl;
        cout << "\n maxl is: " << maxl << endl;
        for (int i = 0; i < nDim; i++)
        {
            for (int j=0; j< nDim; j++)
                {
                        double X = 0; double Y = 0;
//                          cout << "\n leftVec(i) is: " << leftVec(i) << endl;
//                          cout << "\n PMatrix(j,maxl) is: " << PMatrix(j,maxl) << endl;
//                          cout << "\n rightVec(j) is: " << rightVec(j) << endl;
//                          cout << "\n PMatrix(maxr,i) is: " << PMatrix(maxr,i) << endl;
                         
                        X = ((leftVec(i) + DeltaMatrix(i,j)) * PMatrix(j,maxl));
                        Y = ((rightVec(j) + DeltaMatrix(i,j)) * PMatrix(maxr,i));
                        E_task(i,j) = (0.5/nVehicles)*(X + Y);  
                    }
         }
    }
    else if (TaskState == 2)
    {
        for (int i = 0; i < nDim; i++)
            {
            for (int j=0; j< nDim; j++)
                {
                 double X = 0; double Y = 0;
                 double sumaa = 0; double sumbb=0;
                 
                 for (int l=(nVehicles+nTasks); l<nDim ; l++)
                 {sumaa = sumaa + PMatrix(j,l);}
                 for (int l=0; l< nVehicles; l++)
                 {sumbb = sumbb + PMatrix(l,i);}
                        
                 X = (leftVec(i) + DeltaMatrix(i,j)) * sumaa;
                 Y = (rightVec(i) + DeltaMatrix(i,j)) * sumbb;
                 E_task(i,j) = (0.5/nVehicles)*(X + Y);
                 }
            }
    }
    
    else if (TaskState == 3)
    {
        Eigen::VectorXd Ltail(nVehicles);
        Eigen::VectorXd Rhead(nVehicles);
        Ltail = leftVec.tail(nVehicles);
        Rhead = rightVec.head(nVehicles);
        cout << "\n"<< PMatrix << endl;
    
        float LL = 0; float RR = 0;
        LL = Ltail.sum();
        RR = Rhead.sum();

        float L_avg = 0; float R_avg = 0;
        L_avg = LL / nVehicles ;
        R_avg = RR / nVehicles;
        
        cout << "\n Lavg is: " << L_avg << endl;
        cout << "\n Ravg is: " << R_avg << endl;

        for (int i = 0; i < nDim; i++)
            {
            for (int j=0; j< nDim; j++)
                {
                 float X = 0; float Y = 0;
                 float spreadL = 0; float spreadR = 0;
                 float sumaa = 0; float sumbb=0;
                 
                 for (int l=(nVehicles+nTasks); l<nDim ; l++)
                 {sumaa = sumaa + PMatrix(j,l);
                  cout << "\n sumaa is: " << sumaa << endl;
                 }

                 for (int l=0; l< nVehicles; l++)
                 {sumbb = sumbb + PMatrix(l,i);
                  cout << "\n sumbb is: " << sumbb << endl;
                 }

                 for (int l=(nVehicles+nTasks); l<nDim ; l++)
                 {spreadL = round(spreadL + ((leftVec(l) - L_avg))) ;
                 cout << "\n spreadL is: " << spreadL << endl;
                 }

                 for (int l=0; l< nVehicles; l++)
                 {spreadR = round(spreadR + ((rightVec(l) - R_avg))) ;
                  cout << "\n spreadR is: " << spreadR << endl;
                 }

                 cout << "\n leftVec(i) is: " << leftVec(i) << endl;
                 cout << "\n rightVec(i) is: " << rightVec(i) << endl;
                 cout << "\n DeltaMatrix(i,j) is: " << DeltaMatrix(i,j) << endl;

                 X = (leftVec(i) + DeltaMatrix(i,j)) * spreadL * sumaa;
                 Y = (rightVec(i) + DeltaMatrix(i,j)) * spreadR * sumbb;
                 cout << "\n X is: " << X << endl;
                 cout << "\n Y is: " << Y << endl;

                 E_task(i,j) = (0.5/nVehicles)*(X + Y);    
                 }
            }
    }
    
    else
    {cout << "\n No options for Etask given" << endl;}
    cout << "\nE_task is: \n" << E_task << endl;
        
    E_task.leftCols(nVehicles) *= 10000000;
    E_task.bottomRows(nVehicles) *= 10000000;
    E_task.topRightCorner(nVehicles,nVehicles) *= 10000000;  //adding all the constraints for the vehicles and tasks  
    //E_task.topRightCorner(nVehicles,nVehicles) = E_task.bottomLeftCorner(nVehicles,nVehicles).eval();       
    //E_task.topLeftCorner(nVehicles,nVehicles) = E_task.bottomLeftCorner(nVehicles,nVehicles).eval();  
    E_task = E_task/kappa;
    
    if (LoopState == true)
    {
    for (int i = 0; i < nDim; i++)
        {
            for (int j=0; j< nDim; j++)
                {   
                        double lk = PMatrix(j,i) / PMatrix(i,i);	// the "zeroed" Pji
                        if (lk < onemsmall )
                            {lk = lk/(1-lk);} // => the resulting Pji for choice j
                        else
                            {lk = lk0;}  
                        E_loop(i,j) = lk;
                }
        }
    }
    else //E_loop = Trace(P^2(j,i))
    {
        Eigen::MatrixXd sqPMatrix = MatrixXd::Zero(nDim,nDim);
        sqPMatrix = PMatrix * PMatrix;
        for (int i = 0; i < nDim; i++)
        {
            for (int j=0; j< nDim; j++)
                {
                E_loop(i,j) = sqPMatrix(j,i);
                }
        }
    }
    
    E_loop.diagonal().array() = 10000000;
    cout << "\nE_loop is: \n" << E_loop << endl;
    cout << "\nE_task after applying constraints and dividing by kappa is: \n" << E_task << endl;

    E_local = (g * E_loop) + E_task;
    cout << "\n E_local is: \n" << E_local << endl;
    return E_local;
    }
    
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Eigen::MatrixXd UpdateP(Eigen::MatrixXd &UpdatedVMatrix, Eigen::MatrixXd &VMatrix, Eigen::MatrixXd &PMatrix, bool &PState) //calculate Updated P
    {
        Eigen::MatrixXd P = MatrixXd::Zero(nDim,nDim);
        if (PState == true)
        {
            Eigen::MatrixXd dQ = MatrixXd::Zero(nDim,nDim);    
            Eigen::MatrixXd DeltaVMatrix = MatrixXd::Zero(nDim,nDim);
            Eigen::MatrixXd DeltaPMatrix = MatrixXd::Zero(nDim,nDim);
            
            cout << "\n  Previous VMatrix is \n" << VMatrix << endl;
            DeltaVMatrix = (UpdatedVMatrix - VMatrix);
            cout << "\n Delta V Matrix is \n" << DeltaVMatrix << endl;
            Eigen::ColPivHouseholderQR<MatrixXd >lu_decomp(DeltaVMatrix);
            int rank = lu_decomp.rank();
            cout << "\n Rank of Delta V Matrix is: " << rank << endl;
            dQ = DeltaVMatrix * PMatrix;    
            
            cout << "\n dQ:\n" << dQ << endl;
            cout << "\n trace of dQ:\n" << dQ.trace() << endl;
            cout << "\n 1 - (trace of dQ) :\n" << (1 - dQ.trace()) << endl;
            cout << "\n PMatrix old is \n" << PMatrix << endl;             
            
            DeltaPMatrix = (PMatrix * dQ)/(1-(dQ.trace()));
            cout << "\n Delta PMatrix:\n" << DeltaPMatrix << endl;
            UpdatedPMatrix = PMatrix + DeltaPMatrix;
            cout << "\n Updated PMatrix using SM method is:\n" << UpdatedPMatrix << endl;
        }
    else 
        {
            P = (I-UpdatedVMatrix);
            UpdatedPMatrix = P.inverse();
            cout << "\n Updated PMatrix using exact inverse is:\n" << UpdatedPMatrix << endl;
        }           
    return UpdatedPMatrix;
    }

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void NN_algo()
    {             
        //calculating initial values L,R, energy for V and P
         Eigen::MatrixXd E_local = MatrixXd::Zero(nDim,nDim);
         Eigen::MatrixXd UpdatedPMatrix = MatrixXd::Zero(nDim,nDim); 
         E_task = MatrixXd::Zero(nDim,nDim);
         E_loop = MatrixXd::Zero(nDim,nDim);
         vMatBest = MatrixXd::Zero(nDim,nDim);
         Eigen::MatrixXd saveV = MatrixXd::Zero(nDim,nDim);
         UpdatedVMatrix = MatrixXd::Zero(nDim,nDim);

         leftVec = VectorXd(nDim);
         rightVec = VectorXd(nDim);

         std::ofstream outfile3 ("VMatrix");
         std::ofstream outfile4 ("NormVMatrix");
         
         PMatrix = (I - VMatrix).inverse(); //calculate initial P
         cout << "\n initial PMatrix is \n" << PMatrix << endl;
         cout << "\n initial VMatrix is \n" << VMatrix << endl;
         calculateLR(VMatrix, PMatrix); // calculate LR for initial values
         cout << "\n initial leftVec is: \n" << leftVec << endl;
         cout << "\n initial rightVec is: \n" << rightVec << endl;
         
         MatrixXf::Index maxl, maxr;
         double maxleftVecInd = leftVec.maxCoeff(&maxl);
         double maxrightVecInd = rightVec.maxCoeff(&maxr);
        
         kappa = 0.5 * (leftVec(maxl) + rightVec(maxr));
         cout << "\n Kappa is: \n" << kappa << endl;
         
         E_local = calculateE(leftVec,rightVec,PMatrix, TaskState, LoopState); //calculate initial local energy E
         
         cout << "\n initial E_local is: \n" << E_local << endl;
         cout << "\n ////////////////////////////////////////////////////////////////////////// " << endl;
        
         int iteration = 1;
         int FLAG = 1;
         kT = kT_start;

         while (FLAG != 0) //iteration starting 
         {	
            cout << "\n" << iteration << " ITERATION STARTING" << endl;
            cout << "\n kT is " << kT << endl;
            cout << "\n PMatrix before is \n" << PMatrix << endl;
            cout << "\n VMatrix before is \n" << VMatrix << endl;
            
            int est = 0;
            for (int i = 0; i < nDim; i++)
                {                    
                    for (int j = 0; j< nDim; j++)
                    {   
                        if (UpdatedVMatrix(i,j) < 0.7)
                        { est = est +1; 
                        }
                    }
                }
            
            if (est = rDim)
            {
              MatrixXf::Index l,r;
              double lInd = leftVec.maxCoeff(&l);
              double rInd = rightVec.maxCoeff(&r);
              float costValue = 0.5 * (leftVec(l) + rightVec(r));
              if (costValue < costValueBest)
                 {
                 cout << "\n present costValue is " << costValue << endl;
                 cout << "\n best costValue is " << costValueBest << endl;
                 vMatBest = UpdatedVMatrix;
                 costValueBest = costValue;
                 }   
            }
            saveV = VMatrix;
            if (UpdateState == 1)
                {UpdatedVMatrix = syncUpdate(E_local, VMatrix);} //calculate updatedVMatrix using ELocal
            else if(UpdateState == 2)
                {UpdatedVMatrix = asyncRUpdate(E_local, VMatrix);} //calculate updatedVMatrix using ELocal
           
            cout << "\n UpdatedVMatrix now is \n" << UpdatedVMatrix << endl;

            for (int i = 0; i < nDim; i++)
                {                    
                    for (int j = 0; j< nDim; j++)
                    {   
                        if ( std::isnan(UpdatedVMatrix(i,j)) )
                        {
                          cout << "\n The matrix has Nan" << endl;
                          UpdatedVMatrix = saveV;
                          return;
                        }                            
                    }
                }
                
            outfile3 << "\n" << iteration << "\t";
            outfile4 << "\n" << iteration << "\t";
            
            NonNormUpdatedVMatrix = UpdatedVMatrix;
            for (int i = 0; i < nDim; i++)
                {                    
                    for (int j = 0; j< nDim; j++)
                    {   
                        outfile3 << UpdatedVMatrix(i,j) << "\t";
                    }
                }
            
            //Normalising till the values along the row/columns is zero
            cout << "\n ///////////////////////////////////////////////////////////////////// " << endl;
            cout << "\n NORMALISATION BEGINS \n" << endl;
            UpdatedVMatrix = normalisation(UpdatedVMatrix);
            cout << "\n col sum after final normalisation is \n" << UpdatedVMatrix.colwise().sum() << endl;
            cout << "\n row sum after final normalisation is \n" << UpdatedVMatrix.rowwise().sum() << endl;
            cout << "\n Final row and column normalised UpdatedVMatrix is \n" << UpdatedVMatrix << endl;
            cout << "\n NORMALISATION ENDS \n" << endl;
            cout << "\n /////////////////////////////////////////////////////////////////////////////////////// " << endl;
            
            for (int i = 0; i < nDim; i++)
                {                    
                    for (int j = 0; j< nDim; j++)
                    {   
                        outfile4 << UpdatedVMatrix(i,j) << "\t";
                    }
                }
            
            UpdatedPMatrix = UpdateP(UpdatedVMatrix, VMatrix, PMatrix, PState);
            calculateLR(UpdatedVMatrix, UpdatedPMatrix);  
            
            int check = 0;
            for (int i = 0; i < nDim; i++)
                {                    
                    for (int j = 0; j< nDim; j++)
                    {   
                        //if (UpdatedVMatrix(i,j) > 0.9)
                        if ((UpdatedVMatrix(i,j) > 0.9) && (NonNormUpdatedVMatrix(i,j) < 0.9))
                        {
                          check = check+1;
                        }
                    }
                }
                
            if (check == rDim)
            {
                return;
            }
            E_local = calculateE(leftVec,rightVec,UpdatedPMatrix, TaskState, LoopState);
            //cout << "\n E_local is: \n " << E_local << endl;
            kT_fac2 = exp(log(kT_swfac) / (nVehicles * nDim)); // lower T after every neuron update

            cout << "\n kT_fac2 is: \n " << kT_fac2 << endl;

            kT *= kT_fac;
            //kT *= kT_fac2;
            cout << "\n new kT is: " << kT << endl;
            cout << "\n" << iteration << " ITERATION DONE" << endl;
            iteration = iteration + 1;
            VMatrix = UpdatedVMatrix;
            PMatrix = UpdatedPMatrix;
            cout << "\n /*/*/*/*/*/*/*/**/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/ " << endl;
            if (kT < kT_stop) 
            {FLAG = 0;}
            cout << "\n ///////////////////////////////////////////////////////////////////// " << endl;
         }
 }  
 
/////////////////////////////////////////////////////////////////////////////////////////////////

void displaySolution(Eigen::MatrixXd &VMatrix, Eigen::MatrixXd &DeltaMatrix, Eigen::VectorXd &TVec) //parses the solution
    {
        std::string veh_alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        std::string task_alpha = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
        std::string solStrA;
        std::string solStrB;

        int indx = 0;
        int indxB = 0;
        int checkTime;
        cTime = Eigen::VectorXd(nVehicles);
        sub = VectorXd::Ones(nVehicles);

        for (int i = 0; i < nDim; i++)
        {for (int j = 0; j < nDim; j++)
        {
            if(VMatrix(i,j) > 0.55)
                VMatrix(i,j) = 1;
            else if (VMatrix(i,j) < 0.45)
                VMatrix(i,j) = 0;
            else (VMatrix(i,j) = 0.5);
        }
        }
        
        for (int i = 0; i < nVehicles; i++)
        {
            for (int j = 0; j < nDim; j++)
            {if (VMatrix(i,j)==1)
                {indx = j;}
            }
            if (i == 0)
            {
                solStrA = std::string("S") + veh_alpha[i] + std::string(" -> ") + task_alpha[indx-nVehicles];
                solStrB = "max(" + std::to_string(DeltaMatrix(i,indx)) + std::string(" + ") + std::to_string(TVec(indx));
            }
            else
            {
                solStrA = solStrA + std::string(" & S") + veh_alpha[i] + std::string(" -> ") + task_alpha[indx-nVehicles];
                solStrB = solStrB + std::string(", ") + std::to_string(DeltaMatrix(i,indx)) + std::string(" + ") + std::to_string(TVec(indx));
            }
            cTime[i] = DeltaMatrix(i,indx) + TVec(indx);
            
            while (indx <= (nDim-nVehicles-1))
            {
                 for (int j = 0; j < nDim; j++)
                    {if (VMatrix(indx,j)==1)
                        {indxB = j;}
                    }
               
                if (indxB > (nVehicles+nTasks-1))
                {
                    solStrA = solStrA + std::string(" -> E") + veh_alpha[indxB-nVehicles-nTasks];
                    solStrB = solStrB + std::string(" + ") + std::to_string(DeltaMatrix(indx,indxB));
                    cTime[i] = cTime[i] + DeltaMatrix(indx,indxB);
                    solStrB = solStrB + std::string(" = ") + std::to_string(cTime[i]);            
                }
                else
                { 
                     solStrA = solStrA + std::string(" -> ") + task_alpha[indxB-nVehicles];
                     solStrB = solStrB + std::string(" + ") + std::to_string(DeltaMatrix(indx,indxB)) + std::string(" + ") + std::to_string(TVec(indxB));
                     cTime[i] = cTime[i] + DeltaMatrix(indx,indxB) + TVec(indxB);
                }
                indx = indxB;
            }
        }
        solStrB = solStrB + std::string(")");
        checkTimeVec = VectorXd(2*nVehicles);
        checkTimeVec << cTime, cTime + (nVehicles * sub);
        VectorXf::Index maxE;
        checkTime = checkTimeVec.maxCoeff(&maxE);

        cout << "The tasks are ordered as:\n";
        cout << "\n" <<solStrA << endl;
        cout << "\n" <<solStrB << endl;
        cout << "\n" <<checkTime << endl;
    }
    
/////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char* argv[])
    {
    remove("results/Eloop.txt");
    remove("results/Elocal.txt");
    remove("results/Etask.txt");
    remove("results/PMatrix.txt");
    remove("results/DeltaVMatrix.txt");
    remove("results/leftVec.txt");
    remove("results/rightVec.txt");
    remove("VMatrix");
    remove("NormVMatrix");
    remove("V");
    
    nVehicles = atoi(argv[1]);
    nTasks = atoi(argv[2]);
    kT_start = atoi(argv[3]);
    kT_stop = atof(argv[4]);
    kT_fac = atof(argv[5]);
    g = atoi(argv[6]);
    
    if (argc != 12)
    {
        std::cout<<"\n Less input options... exiting "<<endl;
        return(0);
    }

    std::cout<<"\n nVehicles is "<<nVehicles<<endl;
    std::cout<<"\n nTasks is "<<nTasks<<endl;

    nDim = 2*nVehicles + nTasks;; 
    rDim = nVehicles + nTasks;

    std::cout<<"\n nDim is "<<nDim<<endl;
    std::cout<<"\n rDim is "<<rDim<<endl;

    DeltaMatrix = MatrixXd::Ones(nDim,nDim);
    I = MatrixXd::Identity(nDim,nDim); //identity matrix
    TVec = VectorXd(nDim);
           
    if (!strcmp(argv[7],"-random"))
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
     
    else if (!strcmp(argv[7],"-read"))
     {
        ifstream file("tVec.txt");
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
        {cout <<"file not open"<<endl;}
    cout << "\n TVec is: \n" << TVec <<endl;
        ifstream file2("deltaMat.txt");
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
        {cout <<"file not open"<<endl;}
    cout << "\n DeltaMatrix is: \n" << DeltaMatrix <<endl;
    }
    
    else 
    {cout << "\n Invalid option: " << argv[7] << "      exiting....\n";
                return(0);
    }
    
    DeltaMatrix.diagonal().array() = 10000000000;
    DeltaMatrix.leftCols(nVehicles) *= 10000000000;
    DeltaMatrix.bottomRows(nVehicles) *= 10000000000;
    DeltaMatrix.topRightCorner(nVehicles,nVehicles) = DeltaMatrix.bottomLeftCorner(nVehicles,nVehicles).eval();       
    //DeltaMatrix.row(1) += 100* DeltaMatrix.row(0);
    
    std::ofstream outfile1 ("tVec.txt");
    std::ofstream outfile2 ("deltaMat.txt");
    outfile1 << TVec << std::endl;
    outfile2 << DeltaMatrix << std::endl;
    outfile1.close();
    outfile2.close();
    
    cout << "\n Updated DeltaMatrix is: \n" << DeltaMatrix << endl;    
    cout << "\n kT_start is "<< kT_start << endl;
    cout << "\n kT_stop is "<< kT_stop << endl;
    cout << "\n kT_fac is "<< kT_fac << endl;
    cout << "\n gamma is "<< g << endl;
    cout << "\n ///////////////////////////////////////////////////////////////////// " << endl;
    
    if (!strcmp(argv[8],"-SM"))
       {PState = true;}
    else if (!strcmp(argv[8],"-exact"))
       {PState = false;}
    else
       {return(0);}
    
    if (!strcmp(argv[9],"-default"))
       {LoopState = true;}
    else if (!strcmp(argv[9],"-trace"))
       {LoopState = false;}
    else
       {return(0);}
    
    if (!strcmp(argv[10],"-sync"))
       {UpdateState = 1;}
    else if (!strcmp(argv[10],"-async"))
       {UpdateState = 2;}
    else if (!strcmp(argv[10],"-asyncR"))
       {UpdateState = 3;}
    else
       {return(0);}
    
    if (!strcmp(argv[11],"-max"))
       {TaskState = 1;}
    else if (!strcmp(argv[11],"-sum"))
       {TaskState = 2;}
    else if (!strcmp(argv[11],"-spread"))
       {TaskState = 3;}
    else
       {return(0);}
       
    clock_t tStart = clock();
    VMatrix = getVMatrix(nVehicles, nTasks, nDim, rDim); 
    NN_algo();    //NN_algo - updating equations
    cout << "\n Annealing done \n" << endl;

    Gnuplot gp;
    Gnuplot gp2;
    gp << "N = `awk 'NR==2 {print NF}' VMatrix` \n";
    gp << "unset key \n";
    gp << "plot for [i=2:N] 'VMatrix' using 1:i with linespoints" << endl;
//     gp2 << "N = `awk 'NR==2 {print NF}' NormVMatrix` \n";
//     gp2 << "unset key \n";
//     gp2 << "plot for [i=2:N] 'NormVMatrix' using 1:i with linespoints" << endl;
//     
    //thresholding the solution
    for (int i = 0; i < nDim; i++)
        for (int j = 0; j < nDim; j++)
        {
            if(NonNormUpdatedVMatrix(i,j) > 0.55)
                NonNormUpdatedVMatrix(i,j) = 1;
            else if (NonNormUpdatedVMatrix(i,j) < 0.45)
                NonNormUpdatedVMatrix(i,j) = 0;
            else (NonNormUpdatedVMatrix(i,j) = 0.5);
        }
        
    //displaySolution(NonNormUpdatedVMatrix, DeltaMatrix, TVec);//Printing out the solution
    cout << "\n The final solution without normalizing V is: \n" << endl;
    cout << NonNormUpdatedVMatrix << endl;
    cout << "\n ///////////////////////////////////////////////////////////////////// " << endl;

    cout << "\n for THE FINAL SOLUTION AFTER normalisation is: \n" << endl;
    cout << UpdatedVMatrix<<endl;
    displaySolution(UpdatedVMatrix, DeltaMatrix, TVec);//Printing out the solution

    PMatrix = (I-UpdatedVMatrix).inverse();
            for (int i = 0; i < nDim; i++)
                {                    
                    for (int j = 0; j< nDim; j++)
                    {   
                        if (std::isnan(PMatrix(i,j)))
                        {
                          cout << "\nTHIS IS A LOOP SOLUTION" << endl;
                          cout << "\n The last best solution \n" << vMatBest<< endl;
                          displaySolution(vMatBest, DeltaMatrix, TVec);//Printing out the solution
                          return(0);
                        }
                    }
                }
    cout << "\n ///////////////////////////////////////////////////////////////////// " << endl;

    cout << "\n The Best VMatrix during search is: \n" << endl;
    cout << vMatBest <<endl;
    displaySolution(vMatBest, DeltaMatrix, TVec);//Printing out the solution
    cout << "\n ///////////////////////////////////////////////////////////////////// " << endl;

    cout << "\n The given DeltaMatrix is: \n" << endl;
    cout << DeltaMatrix << endl;
    cout << "\nThe input parameters are for : " << argv[7] << " values for finding the " << argv[11] << " cost." << endl;
    printf("\n Total computational time taken: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
}
