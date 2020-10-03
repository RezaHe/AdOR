from utils import *
import numpy as np
import tensorflow as tf
import os


def Loss_MI(preds):
    ScaledPreds = tf.add(preds, tf.stop_gradient(tf.negative(tf.reduce_max(preds))))
    JointData = tf.gather(ScaledPreds, tf.range(0, tf.div(tf.size(ScaledPreds),2), 1))
    MarginalData = tf.gather(ScaledPreds, tf.range(tf.div(tf.size(ScaledPreds),2), tf.size(ScaledPreds), 1))
    return tf.reduce_mean(JointData) - tf.log(tf.reduce_mean(tf.exp(MarginalData)))



class ador(object):
    def __init__(self, NumOfRegressors, BatchSize):
        super(ador, self).__init__()
        
        self.R_BatchSize = BatchSize
        self.R_InputSize = NumOfRegressors
        self.R_LayerIds = []
        self.R_LayerSizes = [NumOfRegressors]
        self.R_LayerActivations = []
        self.R_WeightInitializers = []
        self.R_BiasInitializers = []
        
        self.MI_BatchSize = BatchSize
        self.MI_InputSize = NumOfRegressors + 1
        self.MI_LayerIds = []
        self.MI_LayerSizes = [NumOfRegressors+1]
        self.MI_LayerActivations = []
        self.MI_WeightInitializers = []
        self.MI_BiasInitializers = []
        
        self.AllSumms = ['None']
        self.TrainedModel = None
        self.session = tf.InteractiveSession()
    
    
    
    def Reg_GetDenseLayer(self, LayerID, LayerSize, Activation='leaky_relu', 
                          WeightInitializer=tf.contrib.layers.xavier_initializer(),
                          BiasInitializer=tf.contrib.layers.xavier_initializer()):
        self.R_LayerIds.append("reg/"+LayerID)
        self.R_LayerSizes.append(LayerSize)
        self.R_LayerActivations.append(Activation)
        self.R_WeightInitializers.append(WeightInitializer)
        self.R_BiasInitializers.append(BiasInitializer)
        
    
    
    
    def MI_GetDenseLayer(self, LayerID, LayerSize, Activation='leaky_relu', 
                         WeightInitializer=tf.contrib.layers.xavier_initializer(),
                         BiasInitializer=tf.contrib.layers.xavier_initializer()):
        self.MI_LayerIds.append("mi/"+LayerID)
        self.MI_LayerSizes.append(LayerSize)
        self.MI_LayerActivations.append(Activation)
        self.MI_WeightInitializers.append(WeightInitializer)
        self.MI_BiasInitializers.append(BiasInitializer)
    
    
    
    
        
    def ConstructGraph(self, Reset=1):
        if Reset:
            tf.reset_default_graph()
        
        # placeholder for feeding inputs and related responses
        self.R_Inputs_tf = tf.placeholder(tf.float32, shape=(self.R_BatchSize, self.R_LayerSizes[0]), name='reg_inputs')
        self.R_Labels_tf = tf.placeholder(tf.float32, shape=(self.R_BatchSize, self.R_LayerSizes[-1]), name='reg_labels')
        
        # creating structure of regression network
        h = self._ConstructNN('reg', self.R_Inputs_tf)
        
        self.R_Predictions_tf = tf.identity(h, name='reg_predictions')
        self.R_Error_tf = tf.add(self.R_Labels_tf, tf.negative(self.R_Predictions_tf), name='reg_errors')
        
        self.R_Error_tf_norm = tf.add(self.R_Error_tf, tf.stop_gradient(tf.negative(tf.reduce_mean(self.R_Error_tf))))
        # creating Inputs of MutualInformation network
        JointData = tf.concat([tf.gather(self.R_Inputs_tf, tf.range(0, tf.div(self.R_BatchSize,2), 1), axis=0), 
                               tf.gather(self.R_Error_tf_norm, tf.range(0, tf.div(self.R_BatchSize,2), 1), axis=0)],
                               axis=1, name='joint_input')
        MarginalData = tf.concat([tf.gather(self.R_Inputs_tf, tf.range(0, tf.div(self.R_BatchSize,2), 1), axis=0), 
                                  tf.gather(self.R_Error_tf_norm, tf.range(tf.div(self.R_BatchSize,2), self.R_BatchSize, 1), axis=0)], 
                                  axis=1, name='marginal_input')
        MI_Inputs = tf.concat([JointData, MarginalData], axis=0, name='mi_inputs')
        
        # creating structure of MutualInformation network
        hMI = self._ConstructNN('mi', MI_Inputs)
        self.MIOutput_tf = hMI
        
        
        
    
    
    
    
    
    def _ConstructNN(self, RegOrMI, Input):
        if RegOrMI=='reg':
            LayerIds = self.R_LayerIds
            LayerSizes = self.R_LayerSizes
            WeightInitializers = self.R_WeightInitializers
            BiasInitializers = self.R_BiasInitializers
            LayerActivations = self.R_LayerActivations
            NameOfWeights = 'reg_weights'
            NameOfBiases = 'reg_bias'
        elif RegOrMI=='mi':
            LayerIds = self.MI_LayerIds
            LayerSizes = self.MI_LayerSizes
            WeightInitializers = self.MI_WeightInitializers
            BiasInitializers = self.MI_BiasInitializers
            LayerActivations = self.MI_LayerActivations
            NameOfWeights = 'mi_weights'
            NameOfBiases = 'mi_bias'
        else:
            raise Exception('please specify reg of mi')
            
        
        # Weight and Bias definitions
        for idx, lid in enumerate(LayerIds):
            with tf.variable_scope(lid):
                w = tf.get_variable(NameOfWeights,shape=(LayerSizes[idx], LayerSizes[idx+1]),
                                    initializer=WeightInitializers[idx])
                if 'out' not in lid:
                    b = tf.get_variable(NameOfBiases,shape=(LayerSizes[idx+1]),
                                        initializer=BiasInitializers[idx])
        
        # Calculating Prediction
        h = Input
        for idx, lid in enumerate(LayerIds):
            with tf.variable_scope(lid, reuse=True):
                w = tf.get_variable(NameOfWeights)
                if 'out' in lid:
                    if LayerActivations[idx] is None:
                        h = tf.matmul(h, w, name=lid+'_output')
                    else:
                        PresentAct = LayerActivations[idx]
                        h = PresentAct(tf.matmul(h,w), name=lid+'_output')
                else:
                    b = tf.get_variable(NameOfBiases)
                    if LayerActivations[idx] is None:
                        h = tf.matmul(h, w, name=lid+'_output')
                    else:
                        PresentAct = LayerActivations[idx]
                        h = PresentAct(tf.matmul(h,w)+b, name=lid+'_output')
                    
        return h
        
    
    
    
    
    
    def Reg_DefineOptimizer(self, optimizer=tf.train.AdamOptimizer, **kwargs):
        self.R_LearningRate_tf = tf.placeholder(tf.float32, shape=None, name='reg_learning_rate')
        reg_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="reg")
        # Calculating Loss
        self.R_Loss_tf = tf.abs(Loss_MI(self.MIOutput_tf), name='reg_loss')
        
        # Optimizer
        R_Optimizer = optimizer(self.R_LearningRate_tf, **kwargs)
        
        self.R_grads_and_vars_tf = R_Optimizer.compute_gradients(self.R_Loss_tf, var_list=reg_vars)
        self.R_Loss_Minimize_tf = R_Optimizer.minimize(self.R_Loss_tf, var_list=reg_vars)
        
    
    
    
    
    
    
    def MI_DefineOptimizer(self, optimizer=tf.train.AdamOptimizer, **kwargs):
        self.MI_LearningRate_tf = tf.placeholder(tf.float32, shape=None, name='mi_learning_rate')
        mi_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="mi")
        # Calculating Loss
        self.MI_Loss_tf = tf.negative(Loss_MI(self.MIOutput_tf), name='mi_loss')
        
        # Optimizer
        MI_Optimizer = optimizer(self.MI_LearningRate_tf, **kwargs)
        
        self.MI_grads_and_vars_tf = MI_Optimizer.compute_gradients(self.MI_Loss_tf, var_list=mi_vars)
        self.MI_Loss_Minimize_tf = MI_Optimizer.minimize(self.MI_Loss_tf, var_list=mi_vars)
    
    
    
    
    
    
    
    def TensorboardSummaryCreator(self, directory, Options=['loss_reg', 'loss_mi', 'resid_entr', 'tr_phs', 'ts_phs',
                                                            'all_reg_weights', 'all_mi_weights', 'reg_grad', 'mi_grad']):
        if Options is None:
            return
        else:
            self.AllSumms = Options
            self.SummDir = directory
            pardir = os.path.abspath(os.path.join(directory, os.pardir))
            if not os.path.exists(pardir):
                os.mkdir(pardir)
            if not os.path.exists(directory):
                os.mkdir(directory)
        
        if 'loss_reg' in Options:
            with tf.name_scope('Loss_Regression'):
                self.R_train_tf = tf.placeholder(tf.float32, shape=None, name='train_R')
                summary_R_train = tf.summary.scalar('train_R_summary', self.R_train_tf)
            self.R_summaries_train_tf = tf.summary.merge([summary_R_train])
        
        if 'loss_mi' in Options:
            with tf.name_scope('Loss_MutualInformation'):
                self.MI_train_tf = tf.placeholder(tf.float32, shape=None, name='train_MI')
                summary_MI_train = tf.summary.scalar('train_MI_summary', self.MI_train_tf)
            self.MI_summaries_train_tf = tf.summary.merge([summary_MI_train])
        
        if 'resid_entr' in Options:
            with tf.name_scope('Residual_Entropy'):
                self.Resi_Ent_tf = tf.placeholder(tf.float32,shape=None,name='Res_Ent')
                Resi_Ent_summary = tf.summary.scalar('Res_Ent_summary', self.Resi_Ent_tf)
            self.Entrop_summaries_tf = tf.summary.merge([Resi_Ent_summary])
        
        if 'tr_phs' in Options:
            with tf.name_scope('train_phase'):
                self.tf_mse_tr = tf.placeholder(tf.float32,shape=None,name='mse_summary_train')
                tf_mse_summary_tr = tf.summary.scalar('MSE_train', self.tf_mse_tr)

                self.tf_mae_tr = tf.placeholder(tf.float32,shape=None,name='mae_summary_train')
                tf_mae_summary_tr = tf.summary.scalar('MAE_train', self.tf_mae_tr)
            self.train_summaries = tf.summary.merge([tf_mse_summary_tr, tf_mae_summary_tr])
        
        if 'ts_phs' in Options:
            with tf.name_scope('test_phase'):
                self.tf_mse_ts = tf.placeholder(tf.float32,shape=None,name='mse_summary_test')
                tf_mse_summary_ts = tf.summary.scalar('MSE_Test', self.tf_mse_ts)

                self.tf_mae_ts = tf.placeholder(tf.float32,shape=None,name='mae_summary_test')
                tf_mae_summary_ts = tf.summary.scalar('MAE_test', self.tf_mae_ts)
            self.performance_summaries = tf.summary.merge([tf_mse_summary_ts, tf_mae_summary_ts])
        
        
        if 'all_reg_weights' in Options:
            R_all_weights_summaries = []
            for lid in self.R_LayerIds:
                with tf.name_scope(lid+'_hist'):
                    with tf.variable_scope(lid, reuse=True):
                        w = tf.get_variable('reg_weights')
                        
                        tf_w_hist = tf.summary.histogram('reg_weights_hist', tf.reshape(w,[-1]))
                        if 'out' not in lid:
                            b = tf.get_variable('reg_bias')
                            tf_b_hist = tf.summary.histogram('reg_bias_hist', b)
                            R_all_weights_summaries.extend([tf_w_hist, tf_b_hist])
                        else:
                            R_all_weights_summaries.extend([tf_w_hist])

            self.R_param_summaries = tf.summary.merge(R_all_weights_summaries)
        
        if 'all_mi_weights' in Options:
            MI_all_weights_summaries = []
            for lid in self.MI_LayerIds:
                with tf.name_scope(lid+'_hist'):
                    with tf.variable_scope(lid, reuse=True):
                        w = tf.get_variable('mi_weights')
                        
                        tf_w_hist = tf.summary.histogram('mi_weights_hist', tf.reshape(w,[-1]))
                        if 'out' not in lid:
                            b = tf.get_variable('mi_bias')
                            tf_b_hist = tf.summary.histogram('mi_bias_hist', b)
                            MI_all_weights_summaries.extend([tf_w_hist, tf_b_hist])
                        else:
                            MI_all_weights_summaries.extend([tf_w_hist])

            self.MI_param_summaries = tf.summary.merge(MI_all_weights_summaries)
        
        
        if 'reg_grad' in Options:
            for g,v in self.R_grads_and_vars_tf:
                if 'out' in v.name and 'weights' in v.name:
                    with tf.name_scope('reg_gradients'):
                        tf_last_grad_norm = tf.sqrt(tf.reduce_mean(tf.pow(g,2)))
                        self.R_gradnorm_summary = tf.summary.scalar('reg_grad_norm', tf_last_grad_norm)
                        break
        
        if 'mi_grad' in Options:
            for g,v in self.MI_grads_and_vars_tf:
                if 'out' in v.name and 'weights' in v.name:
                    with tf.name_scope('mi_gradients'):
                        tf_last_grad_norm = tf.sqrt(tf.reduce_mean(tf.pow(g,2)))
                        self.MI_gradnorm_summary = tf.summary.scalar('mi_grad_norm', tf_last_grad_norm)
                        break
                        
    
    
    
    
    def GetSession(self, config, Reset=1):
        if Reset:
            self.session.close()
        self.session = tf.InteractiveSession(config=config)
        
        if not 'None' in self.AllSumms:
            self.summ_writer = tf.summary.FileWriter(self.SummDir, self.session.graph)
        
        self.session.run(tf.global_variables_initializer())
    
    
    
    
    
    # =================================== Training of Regression ========================================
    def UpdateRegNet(self, X, Y, iteration, RegLearnRate, nReg_Steps):
        n_samples = np.shape(X)[0]
        train_batch_Indeces = batch_index_generator(n_samples, self.R_BatchSize, nReg_Steps)
        
        R_loss_per_iter = []
        MSE_per_iter_tr = []
        MAE_per_iter_tr = []
        
        for i in range(nReg_Steps):
            R_batch_train_inputs = X[train_batch_Indeces[:,i],:]
            R_batch_train_labels = Y[train_batch_Indeces[:,i],:]
            
            # calling loss-minimizer and updating weights and biases
            R_l,_,R_train_batch_predictions = self.session.run([self.R_Loss_tf, self.R_Loss_Minimize_tf, self.R_Predictions_tf],
                                                          feed_dict={self.R_Inputs_tf: R_batch_train_inputs,
                                                                     self.R_Labels_tf: R_batch_train_labels,
                                                                     self.R_LearningRate_tf: RegLearnRate})
            
            if i == 0:
                if 'all_reg_weights' in self.AllSumms:
                    R_wb_summ = self.session.run(self.R_param_summaries,feed_dict={self.R_Inputs_tf: R_batch_train_inputs,
                                                                                  self.R_Labels_tf: R_batch_train_labels,
                                                                                  self.R_LearningRate_tf: RegLearnRate})
                    self.summ_writer.add_summary(R_wb_summ, iteration)
                if 'reg_grad' in self.AllSumms:
                    R_gn_summ = self.session.run(self.R_gradnorm_summary,feed_dict={self.R_Inputs_tf: R_batch_train_inputs,
                                                                                   self.R_Labels_tf: R_batch_train_labels,
                                                                                   self.R_LearningRate_tf: RegLearnRate})
                    self.summ_writer.add_summary(R_gn_summ, iteration)
                
            R_loss_per_iter.append(R_l)
            MSE_per_iter_tr.append(accuracy_MSE(R_train_batch_predictions, R_batch_train_labels))
            MAE_per_iter_tr.append(accuracy_MAE(R_train_batch_predictions, R_batch_train_labels))
            
        # updating Predictions
        AllPredicts = self.PredictAfterTrain(X)
        
        Residuals = AllPredicts - Y
        
        R_avg_loss = np.mean(R_loss_per_iter)
        avg_MSE_tr = np.mean(MSE_per_iter_tr)
        avg_MAE_tr = np.mean(MAE_per_iter_tr)
        Res_Ent_Per = kNNEntropy(Residuals)
        
        if 'loss_reg' in self.AllSumms:
            summ_loss_R = self.session.run(self.R_summaries_train_tf, feed_dict={self.R_train_tf: R_avg_loss})
            self.summ_writer.add_summary(summ_loss_R, iteration)
        if 'tr_phs' in self.AllSumms:
            summ_train = self.session.run(self.train_summaries, feed_dict={self.tf_mse_tr: avg_MSE_tr,
                                                                           self.tf_mae_tr: avg_MAE_tr})
            self.summ_writer.add_summary(summ_train, iteration)
        if 'resid_entr' in self.AllSumms:
            summ_ent = self.session.run(self.Entrop_summaries_tf, feed_dict={self.Resi_Ent_tf: Res_Ent_Per})
            self.summ_writer.add_summary(summ_ent, iteration)
        
        return AllPredicts, Residuals, R_avg_loss, avg_MAE_tr, avg_MSE_tr, Res_Ent_Per
    
    
    
    
    # =================================== Training of MutualInformation ========================================
    def UpdateMINet(self, X, Y, iteration, MILearnRate, nMI_Steps):
        n_samples = np.shape(X)[0]
        
        train_batch_Indeces = batch_index_generator(n_samples, self.MI_BatchSize, nMI_Steps)
        
        MI_loss_per_iter = []
        
        for i in range(nMI_Steps):
            MI_batch_train_inputs = X[train_batch_Indeces[:,i],:]
            MI_batch_train_labels = Y[train_batch_Indeces[:,i],:]

            MI_l,_ = self.session.run([self.MI_Loss_tf, self.MI_Loss_Minimize_tf],feed_dict={self.R_Inputs_tf: MI_batch_train_inputs,
                                                                                             self.R_Labels_tf: MI_batch_train_labels,
                                                                                             self.MI_LearningRate_tf: MILearnRate})
            if i == 0:
                if 'all_mi_weights' in self.AllSumms:
                    MI_wb_summ = self.session.run(self.MI_param_summaries,feed_dict={self.R_Inputs_tf: MI_batch_train_inputs,
                                                                                  self.R_Labels_tf: MI_batch_train_labels,
                                                                                  self.MI_LearningRate_tf: MILearnRate})
                    self.summ_writer.add_summary(MI_wb_summ, iteration)
                if 'mi_grad' in self.AllSumms:
                    MI_gn_summ = self.session.run(self.MI_gradnorm_summary,feed_dict={self.R_Inputs_tf: MI_batch_train_inputs,
                                                                                   self.R_Labels_tf: MI_batch_train_labels,
                                                                                   self.MI_LearningRate_tf: MILearnRate})
                    self.summ_writer.add_summary(MI_gn_summ, iteration)
            
            MI_loss_per_iter.append(MI_l)
        
        MI_avg_loss = np.mean(MI_loss_per_iter)
        if 'loss_mi' in self.AllSumms:
            summ_loss_MI = self.session.run(self.MI_summaries_train_tf, feed_dict={self.MI_train_tf: MI_avg_loss})
            self.summ_writer.add_summary(summ_loss_MI, iteration)
        
        return MI_avg_loss
    
    
    
    
    
    
    def TestModel(self, X, Y, iteration):
        n_samples = np.shape(X)[0]
        
        test_batch_Indeces = batch_index_generator(n_samples, self.MI_BatchSize, 1)
        
        batch_test_inputs = X[test_batch_Indeces[:,0],:]
        batch_test_labels = Y[test_batch_Indeces[:,0],:]
        
        test_batch_predictions = self.session.run(self.R_Predictions_tf, feed_dict={self.R_Inputs_tf: batch_test_inputs})
        
        MSE_per_iter_ts = accuracy_MSE(test_batch_predictions, batch_test_labels)
        MAE_per_iter_ts = accuracy_MAE(test_batch_predictions, batch_test_labels)
        if 'ts_phs' in self.AllSumms:
            summ_test = self.session.run(self.performance_summaries, feed_dict={self.tf_mse_ts: MSE_per_iter_ts,
                                                                                self.tf_mae_ts: MAE_per_iter_ts})
            self.summ_writer.add_summary(summ_test, iteration)
        
        return MSE_per_iter_ts, MAE_per_iter_ts
    
    
    
    
    
    
    
    
    def PredictAfterTrain(self, X):
        n_samples = np.shape(X)[0]
        NumZeroPad = self.R_BatchSize - n_samples%self.R_BatchSize
        X_ext = np.concatenate([X, np.zeros(shape=(NumZeroPad, np.shape(X)[1]))], axis=0)
        n_samples_Ext = np.shape(X_ext)[0]
        Indeces = batch_index_generator(n_samples_Ext, self.R_BatchSize, np.int32(n_samples_Ext/self.R_BatchSize))
        
        
        WholePreds = np.zeros(shape=(n_samples_Ext, 1))
        for i in range(np.int32(n_samples_Ext/self.R_BatchSize)):
            InFeed = X_ext[Indeces[:,i],:]
            R_preds = self.session.run([self.R_Predictions_tf],
                                       feed_dict={self.R_Inputs_tf: InFeed})
            WholePreds[Indeces[:,i],:] = R_preds
        
        return WholePreds[:-NumZeroPad, :]
    
    
    
    
    def EndSession(self, CloseSess=1):
              
        if CloseSess:
            self.session.close()
            