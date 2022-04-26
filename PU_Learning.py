import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Tuple
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score, mean_squared_error as mse
from sklearn.linear_model import LogisticRegression
import math

class TwoStepTechnique(BaseEstimator, ABC):
    
    def __init__(self):
        self.classifier=None
    
    @abstractmethod
    def step1(self, X, s) -> Tuple[np.ndarray, np.ndarray]:
        
        pass
    
    @abstractmethod
    def step2(self, X, n, p)->BaseEstimator:

        pass
    
    def fit(self, X, s):
        ## ADD YOUR CODE HERE
        p,n,u=self.step1(X,s)
        self.classifier=self.step2( X, p, n,u)
        return self
    
    def predict(self,X):
        ## ADD YOUR CODE HERE
        prediction = self.classifier.predict(X)
        return prediction
    
    def predict_proba(self, X):
        ## ADD YOUR CODE HERE
        probability = self.classifier.predict_proba(X)
        return probability

class SEM(TwoStepTechnique):
    
    def __init__(self,
                 tol = 1.0e-10,
                 max_iter = 100,
                 spy_prop = 0.5,
                 l = 0.15,
                 classifier_step_1 = LogisticRegression(),
                 classifier_step_2= LogisticRegression(),
                 seed=331
                ):
        
        super().__init__()
        
        # instantiate the parameters
        self.tol=tol
        self.max_iter=max_iter
        self.spy_prop=spy_prop
        self.l=l
        self.classifier_step_1=classifier_step_1
        self.classifier_step_2=classifier_step_2
        self.seed=seed
        ## ADD YOUR CODE HERE
        
    def step1(self, X, s) -> Tuple[np.ndarray, np.ndarray]:
        
        np.random.seed(self.seed)
      
        # Split the dataset into P (Positive) and M (Mix of positives and negatives)
        P = X[s == 1]
        M= X[s==0]
       
        

        
        # Select (randomly) the spies S
        spie_mask = np.random.random(s.sum()) < self.spy_prop
        MS = np.vstack([X[s == 0], X[s == 1][spie_mask]])        
        MS_spies = np.hstack([np.zeros((s == 0).sum()), np.ones(spie_mask.sum())])
        
        # Positive with spies removed
        P = X[s == 1][~spie_mask]
        # Combo
        MSP = np.vstack([MS, P])
        
        # Labels
        MSP_y = np.hstack([np.zeros(MS.shape[0]), np.ones(P.shape[0])])
        
        # Fit first model
        
        
        
        # Update P and MS
        
    
        
        ### I-EM Algorithm

        # Train the classifier using P and MS:
       
        self.classifier_step_1.fit(MSP, MSP_y)
        score_variation= self.classifier_step_1.score(MSP, MSP_y)
      
        prob= self.classifier_step_1.predict_proba(MS)[:,1]
        
        
        # Save the model's score ''score_variation'' using model.score function
        
        
        # Initialize iterations to 0 and the score variation
        n_iter=0
        
        #Loop while classifier parameters change, i.e. until the score variation is >= tolerance
        while score_variation >= self.tol and n_iter < self.max_iter:
            new=np.hstack([prob, np.ones(P.shape[0])])
           
            
            
            
            
            # Expectation step
           
            # Create the new training set with the probabilistic labels (weights)
            

            
            #Maximization step
            self.classifier_step_1.fit(MSP, MSP_y,new)
            prob=self.classifier_step_1.predict_proba(MS)[:,1]
            n_iter=n_iter+1
            
            #Update score variation and the old score
            score_variation= self.classifier_step_1.score(MSP, MSP_y)
            
        # Print the number of iterations as sanity check
        print("Number of iterations first step:", n_iter)
        
        # Select the threshold t such that l% of spies' probabilities to be positive is belot t
        A=[]
        ##Checking for the weights of the Spies
        for i in range(0,len(MS_spies)):
          if(MS_spies[i]==1):
              
               A.append(new[i])
        A=sorted(A)
        ##Checking for the threshold value according to the L value given        
        Values= len(A) * self.l
        Values=math.ceil(Values)
       
        Threshold=A[Values-1]
        
        # Create N and U
        N=[]
        U=[]
        j=0
        k=0
        for i in range(0,len(new)):
          if(new[i]<Threshold):
               b=MSP[i].tolist()
               N.append(b)
          else:
            if(new[i]<1 and new[i]>Threshold):
               c=MSP[i].tolist()
               U.append(c)
           
        ##Putting spies back into P
        P = X[s == 1]
              
        # Return P, N, U
        return P, N, U
        
    
    def step2(self, X, P, N, U)->BaseEstimator:
        np.random.seed(self.seed)
        
        # Assign every document in the positive set pos the fixed class label 1

        # Assign every document in the likely negative set N the initial class label 0
        ##Stacking Positive and Likely Negative for initial classifier training
        NP=np.vstack([N,P])
    
        NP_y = np.hstack([np.zeros(len(N)), np.ones(P.shape[0])])
   
        ###I-EM Algorithm
      

        # Train classifier using M and P:
        self.classifier_step_2.fit(NP, NP_y)
        ##Providing weights to Unlabeled set so that they can participate in EM now
        prob=self.classifier_step_2.predict_proba(U)[:,1]
        for i in range(0,len(prob)):
          if(prob[i]>0.5):
            prob[i]=1
          else:
           if(prob[i]<=0.5):
             prob[i]=0  
     
        # Compute the metrics for classifier f_i in delta_i to select the best classifier

        score_variation= self.classifier_step_2.score(NP, NP_y)
       
        UN=np.vstack([U,N])
        UN_y=np.hstack([prob, np.zeros(len(N))])
       
       
        # Initialize iterations to 0, the score variation, and whether the best classifier has been selected or not.
        n_iter=0
        Probval=self.classifier_step_2.predict_proba(UN)[:,1]
        self.final_classifier=self.classifier_step_2
        # Loop until the variation is > than the tolerance
        while score_variation >= self.tol and n_iter < self.max_iter:

            
            # Update probabilities
            
            
            
            # Create the new training set with the probabilistic labels (weights)

            
            
            #Maximization step

            self.classifier_step_2.fit(UN, UN_y,Probval)
            Probval=self.classifier_step_2.predict_proba(UN)[:,1]
            n_iter=n_iter+1
            #Update parameter variation
            #Select the best classifier classifier: (final_classifier)

            if(self.classifier_step_2.score(UN, UN_y) > score_variation):
              self.final_classifier=self.classifier_step_2
            
            

            

        print("Number of iterations second step:", n_iter)
        
        
        return self.final_classifier

    
    
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Tuple
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score, mean_squared_error as mse
from sklearn.linear_model import LogisticRegression
          
    
#--------------------#--------------------#--------------------#--------------------    
#-------------------- Second PU Learning Method #--------------------


    
class ModifiedLogisticRegression(BaseEstimator):

    def __init__(self,
                 tol = 1.0e-10,
                 max_iter = 100,
                 l_rate = 0.001,
                 c = 0,
                 seed = 331):
        self.tol=tol
        self.max_iter=max_iter
        self.l_rate=l_rate
        self.c=c
        self.seed=seed
        self.w = 0
        self.b = 0.2
        self.FinalClassifier = LogisticRegression()
        
        #instantiate the parameters
        ## ADD YOUR CODE HERE
        
    def log_likelihood(self, x, y):
        #If you use the gradient ascent technique, fill in this part with the log_likelihood function
        #If you use EM method or a different technique, you can leave this empty
        for i in range(0,len(x)):
          like = np.mean((np.log(1/(1+self.b*self.b+np.exp(-self.w*x[i]))) + (1-y[i])*np.log(1-(1/(1+self.b*self.b+np.exp(-self.w*x[i]))))))
          
        
        return like
        
    def parameters_update(self, x, y):
        #If you use the gradient ascent technique, fill in this part with the parameter update (both w and b)
        #If you use EM method or a different technique, you can leave this empty
        for i in range(0,len(x)):
          ft=1+(self.b*self.b)+np.exp(-self.w*x[i])
          st=(self.b*self.b)+ np.exp(-self.w*x[i])
          dw=np.mean((x[i]*np.exp(-self.w*x[i]))*((y[i]/st)-(1/(ft*st))))
          if(math.isnan(dw)):
            dw=np.mean((x[i-1]*np.exp(-self.w*x[i-1]))*((y[i-1]/st)-(1/(ft*st))))
          
        # dw=np.mean( (x[i]*np.exp(-self.w*x[i]))*( y[i]/( self.b**2 + np.exp(-self.w*x[i]) ) - (1/( 1 + self.b**2 + np.exp(-self.w*x[i]) ) * (self.b**2 + np.exp(self.w*x[i]))) ))
         #dw=np.mean(x[i]*np.exp(-self.w*x[i])*((y[i]/self.b*self.b+np.exp(-self.w*x[i]))-(1/(1+self.b*self.b+np.exp(-self.w*x[i]))(self.b*self.b+np.exp(-self.w*x[i])))))
       #  db=np.mean(2*self.b*(1-y[i]*(1+self.b*self.b+np.exp(-self.w*x[i]))/(1+self.b*self.b+np.exp(-self.w*x[i]))(self.b*self.b+np.exp(-self.w*x[i]))))
          db=np.mean((2*self.b)*((1-y[i]*(1+self.b**2*np.exp(-self.w*x[i])))/( 1 + self.b**2 + np.exp(-self.w*x[i]) ) * (self.b**2 + np.exp(self.w*x[i]))))        
          if(math.isnan(db)):
            db=1
          
        w_new=self.w+self.l_rate*dw
        b_new=self.b+self.l_rate*db
        
        self.w=w_new
        self.b=b_new
        
        return w_new,b_new
    
    def fit(self, X, s):
        np.random.seed(self.seed)
        m, n = X.shape
        #initialize w and b
        self.w = np.zeros((n,1))
        s = s.reshape(m,1)
        likelihood = []
        epochs=100
        bs=50

  
        #inizialize the score (log_likelihood), the number of iterations and the score variation.
        log_likelihoo=0
        n_iter=0
        score_variation=1
        for epoch in range(epochs):
          for i in range((m-1)//bs + 1):

             start_i = i*bs
             end_i = start_i + bs
             xb = X[start_i:end_i]
             yb = s[start_i:end_i] 
        #loop until the score variation is lower than tolerance or max_iter is reached
             like=self.log_likelihood(xb,yb)
            #Maximization step (update the parameters)
             self.w,self.b=self.parameters_update(xb,yb)
             
             
            
             
            #Expectation step (compute log_likelihood)
             
            
            #update scores
    
        
        self.c=1/(1+self.b**2)
        Classifier = LogisticRegression()
        Classifier.fit(X, s)
        val=Classifier.predict_proba(X)[:,1]
       
        final=val/self.c
     
     ##Rounding off the predict_proba values 
        for j in range(0,len(final)):
         if (final[j]>0.5):
          final[j] = 1
         else:
          final[j] = 0
       
        self.FinalClassifier.fit(X,final)

            
        return self    
    
    def estimate_c(self):
        # Estimate the parameter c from b
        self.c=1/(1+self.b**2)
        
        
       # val=Classifier.predict_proba(self.X)
      #  print(val)

        ## ADD YOUR CODE HERE
        return
    
    def predict(self,X):
        ## ADD YOUR CODE HERE
        prediction=self.FinalClassifier.predict(X)
        return prediction
    
    def predict_proba(self, X):
        ## ADD YOUR CODE HERE
        probability=self.FinalClassifier.predict_proba(X)
        return probability
    
    
    
    
    
    
    
    