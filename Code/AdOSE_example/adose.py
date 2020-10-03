from utils import *
import numpy as np
import tensorflow as tf
import os



def Loss_KL(preds):
    ScaledPreds = tf.add(preds, tf.stop_gradient(tf.negative(tf.reduce_max(preds))))
    Dist1 = tf.gather(ScaledPreds, tf.range(0, tf.div(tf.size(ScaledPreds),2), 1))
    Dist2 = tf.gather(ScaledPreds, tf.range(tf.div(tf.size(ScaledPreds),2), tf.size(ScaledPreds), 1))
    return tf.reduce_mean(Dist1) - tf.log(tf.reduce_mean(tf.exp(Dist2)))



class adose(object):
    def __init__(self, NumOfRegressors, BatchSize):
        super(adose, self).__init__()
        
        self.R_BatchSize = BatchSize
        self.R_InputSize = NumOfRegressors + 1
        self.R_LayerIds = []
        self.R_LayerSizes = [NumOfRegressors+1]
        self.R_LayerActivations = []
        self.R_WeightInitializers = []
        self.R_BiasInitializers = []
        
        self.RanGen_BatchSize = BatchSize
        self.RanGen_InputSize = 1
        self.RanGen_LayerIds = []
        self.RanGen_LayerSizes = [1]
        self.RanGen_LayerActivations = []
        self.RanGen_WeightInitializers = []
        self.RanGen_BiasInitializers = []
        
        self.KL_BatchSize = 2*BatchSize
        self.KL_InputSize = NumOfRegressors + 1
        self.KL_LayerIds = []
        self.KL_LayerSizes = [NumOfRegressors+1]
        self.KL_LayerActivations = []
        self.KL_WeightInitializers = []
        self.KL_BiasInitializers = []
        
        self.AllSumms = ['None']
        self.TrainedModel = None
        self.session = tf.InteractiveSession()
    
    
    
    
    
    
    def RanGen_GetDenseLayer(self, LayerID, LayerSize, Activation='leaky_relu', 
                          WeightInitializer=tf.contrib.layers.xavier_initializer(),
                          BiasInitializer=tf.contrib.layers.xavier_initializer()):
        self.RanGen_LayerIds.append("reg/rangen/"+LayerID)
        self.RanGen_LayerSizes.append(LayerSize)
        self.RanGen_LayerActivations.append(Activation)
        self.RanGen_WeightInitializers.append(WeightInitializer)
        self.RanGen_BiasInitializers.append(BiasInitializer)
    
    
    
    
    
    def Reg_GetDenseLayer(self, LayerID, LayerSize, Activation='leaky_relu', 
                          WeightInitializer=tf.contrib.layers.xavier_initializer(),
                          BiasInitializer=tf.contrib.layers.xavier_initializer()):
        self.R_LayerIds.append("reg/"+LayerID)
        self.R_LayerSizes.append(LayerSize)
        self.R_LayerActivations.append(Activation)
        self.R_WeightInitializers.append(WeightInitializer)
        self.R_BiasInitializers.append(BiasInitializer)
        
    
    
    
    def KL_GetDenseLayer(self, LayerID, LayerSize, Activation='leaky_relu', 
                         WeightInitializer=tf.contrib.layers.xavier_initializer(),
                         BiasInitializer=tf.contrib.layers.xavier_initializer()):
        self.KL_LayerIds.append("kl/"+LayerID)
        self.KL_LayerSizes.append(LayerSize)
        self.KL_LayerActivations.append(Activation)
        self.KL_WeightInitializers.append(WeightInitializer)
        self.KL_BiasInitializers.append(BiasInitializer)
    
    
    
    
        
    def ConstructGraph(self, Reset=1):
        if Reset:
            tf.reset_default_graph()
        # placeholder for feeding random inputs
        self.RanGen_Inputs_tf = tf.placeholder(tf.float32, shape=(self.RanGen_BatchSize, 1), name='reg_rangen_inputs')
        
        # creating structure of distribution network
        hRan = self._ConstructNN('rangen', self.RanGen_Inputs_tf)
        
        self.RanDist = tf.identity(hRan, name='reg_rangen_distribution')
        
        
        
        # placeholder for feeding inputs and related responses
        self.R_Inputs_tf = tf.placeholder(tf.float32, shape=(self.R_BatchSize, self.R_InputSize-1), name='reg_inputs')
        self.R_Labels_tf = tf.placeholder(tf.float32, shape=(self.R_BatchSize, self.R_LayerSizes[-1]), name='reg_labels')
        
        # concatinate inputs and random-number
        R_Inputs_Randomized = tf.concat([self.R_Inputs_tf, self.RanDist], axis=1, name='reg_randomized_inputs')
        
        # creating structure of regression network
        h = self._ConstructNN('reg', R_Inputs_Randomized)
        
        self.R_Predictions_tf = tf.identity(h, name='reg_predictions')
        self.R_Error_tf = tf.add(self.R_Labels_tf, tf.negative(self.R_Predictions_tf), name='reg_errors')
        
        
        
        # creating Inputs of KL-Distance network
        PData = tf.concat([self.R_Labels_tf, self.R_Inputs_tf], axis=1, name='first_dist')
        QData = tf.concat([self.R_Predictions_tf, self.R_Inputs_tf], axis=1, name='sec_dist')
        KL_Inputs = tf.concat([PData, QData], axis=0, name='kl_inputs')
        
        # creating structure of MutualInformation network
        hKL = self._ConstructNN('kl', KL_Inputs)
        self.KLOutput_tf = hKL
        
        
        
    
    
    
    
    
    def _ConstructNN(self, RegOrKLOrRangen, Input):
        if RegOrKLOrRangen=='reg':
            LayerIds = self.R_LayerIds
            LayerSizes = self.R_LayerSizes
            WeightInitializers = self.R_WeightInitializers
            BiasInitializers = self.R_BiasInitializers
            LayerActivations = self.R_LayerActivations
            NameOfWeights = 'reg_weights'
            NameOfBiases = 'reg_bias'
        elif RegOrKLOrRangen=='kl':
            LayerIds = self.KL_LayerIds
            LayerSizes = self.KL_LayerSizes
            WeightInitializers = self.KL_WeightInitializers
            BiasInitializers = self.KL_BiasInitializers
            LayerActivations = self.KL_LayerActivations
            NameOfWeights = 'kl_weights'
            NameOfBiases = 'kl_bias'
        elif RegOrKLOrRangen=='rangen':
            LayerIds = self.RanGen_LayerIds
            LayerSizes = self.RanGen_LayerSizes
            WeightInitializers = self.RanGen_WeightInitializers
            BiasInitializers = self.RanGen_BiasInitializers
            LayerActivations = self.RanGen_LayerActivations
            NameOfWeights = 'rangen_weights'
            NameOfBiases = 'rangen_bias'
        else:
            raise Exception('please specify reg or kl or rangen')
            
        
        # Weight and Bias definitions
        for idx, lid in enumerate(LayerIds):
            with tf.variable_scope(lid):
                w = tf.get_variable(NameOfWeights,shape=(LayerSizes[idx], LayerSizes[idx+1]),
                                    initializer=WeightInitializers[idx])
                if 'out' not in lid:
                    b = tf.get_variable(NameOfBiases,shape= (LayerSizes[idx+1]),
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
        self.reg_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="reg")
        # Calculating Loss
        self.R_Loss_tf = tf.identity(Loss_KL(self.KLOutput_tf), name='reg_loss')
        
        # Optimizer
        R_Optimizer = optimizer(self.R_LearningRate_tf, **kwargs)
        
        self.R_grads_and_vars_tf = R_Optimizer.compute_gradients(self.R_Loss_tf, var_list=self.reg_vars)
        self.R_Loss_Minimize_tf = R_Optimizer.minimize(self.R_Loss_tf, var_list=self.reg_vars)
        
    
    
    
    
    
    
    def KL_DefineOptimizer(self, optimizer=tf.train.AdamOptimizer, **kwargs):
        self.KL_LearningRate_tf = tf.placeholder(tf.float32, shape=None, name='kl_learning_rate')
        self.kl_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="kl")
        # Calculating Loss
        self.KL_Loss_tf = tf.negative(Loss_KL(self.KLOutput_tf), name='kl_loss')
        
        # Optimizer
        KL_Optimizer = optimizer(self.KL_LearningRate_tf, **kwargs)
        
        self.KL_grads_and_vars_tf = KL_Optimizer.compute_gradients(self.KL_Loss_tf, var_list=self.kl_vars)
        self.KL_Loss_Minimize_tf = KL_Optimizer.minimize(self.KL_Loss_tf, var_list=self.kl_vars)
    
    
    
    
    
    
    
    def TensorboardSummaryCreator(self, directory, Options=['loss_reg', 'loss_kl', 'resid_entr', 'tr_phs', 'ts_phs', 'all_reg_weights',
                                                            'all_rangen_weights', 'all_kl_weights', 'reg_grad', 'kl_grad']):
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
        
        if 'loss_kl' in Options:
            with tf.name_scope('Loss_KLDistance'):
                self.KL_train_tf = tf.placeholder(tf.float32, shape=None, name='train_KL')
                summary_KL_train = tf.summary.scalar('train_KL_summary', self.KL_train_tf)
            self.KL_summaries_train_tf = tf.summary.merge([summary_KL_train])
        
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
        
        
        if 'all_rangen_weights' in Options:
            Rangen_all_weights_summaries = []
            for lid in self.RanGen_LayerIds:
                with tf.name_scope(lid+'_hist'):
                    with tf.variable_scope(lid, reuse=True):
                        w = tf.get_variable('rangen_weights')
                        
                        tf_w_hist = tf.summary.histogram('rangen_weights_hist', tf.reshape(w,[-1]))
                        if 'out' not in lid:
                            b = tf.get_variable('rangen_bias')
                            tf_b_hist = tf.summary.histogram('rangen_bias_hist', b)
                            Rangen_all_weights_summaries.extend([tf_w_hist, tf_b_hist])
                        else:
                            Rangen_all_weights_summaries.extend([tf_w_hist])
            
            self.Rangen_param_summaries = tf.summary.merge(Rangen_all_weights_summaries)
        
        
        if 'all_kl_weights' in Options:
            KL_all_weights_summaries = []
            for lid in self.KL_LayerIds:
                with tf.name_scope(lid+'_hist'):
                    with tf.variable_scope(lid, reuse=True):
                        w = tf.get_variable('kl_weights')
                        
                        tf_w_hist = tf.summary.histogram('kl_weights_hist', tf.reshape(w,[-1]))
                        if 'out' not in lid:
                            b = tf.get_variable('kl_bias')
                            tf_b_hist = tf.summary.histogram('kl_bias_hist', b)
                            KL_all_weights_summaries.extend([tf_w_hist, tf_b_hist])
                        else:
                            KL_all_weights_summaries.extend([tf_w_hist])

            self.KL_param_summaries = tf.summary.merge(KL_all_weights_summaries)
        
        
        if 'reg_grad' in Options:
            for g,v in self.R_grads_and_vars_tf:
                if 'out' in v.name and 'weights' in v.name:
                    with tf.name_scope('reg_gradients'):
                        tf_last_grad_norm = tf.sqrt(tf.reduce_mean(tf.pow(g,2)))
                        self.R_gradnorm_summary = tf.summary.scalar('reg_grad_norm', tf_last_grad_norm)
                        break
        
        if 'kl_grad' in Options:
            for g,v in self.KL_grads_and_vars_tf:
                if 'out' in v.name and 'weights' in v.name:
                    with tf.name_scope('kl_gradients'):
                        tf_last_grad_norm = tf.sqrt(tf.reduce_mean(tf.pow(g,2)))
                        self.KL_gradnorm_summary = tf.summary.scalar('kl_grad_norm', tf_last_grad_norm)
                        break
                        
    
    
    
    
    def GetSession(self, config, Reset=1):
        if Reset:
            self.session.close()
        self.session = tf.InteractiveSession(config=config)
        
        if not 'None' in self.AllSumms:
            self.summ_writer = tf.summary.FileWriter(self.SummDir, self.session.graph)
        
        self.session.run(tf.global_variables_initializer())
    
    
    
    
    
    
    # =================================== Training of Regression ========================================
    def UpdateRegNet(self, X, Y, RandomIn, iteration, RegLearnRate, nReg_Steps):
        n_samples = np.shape(X)[0]
        train_batch_Indeces = batch_index_generator(n_samples, self.R_BatchSize, nReg_Steps)
        
        R_loss_per_iter = []
        MSE_per_iter_tr = []
        MAE_per_iter_tr = []
        
        for i in range(nReg_Steps):
            R_batch_train_inputs = X[train_batch_Indeces[:,i],:]
            R_batch_train_labels = Y[train_batch_Indeces[:,i],:]
            RandomIn = np.random.permutation(RandomIn)
            Ran_batch_train = RandomIn[train_batch_Indeces[:,i],:]
            
            # calling loss-minimizer and updating weights and biases
            R_l,_,R_train_batch_predictions = self.session.run([self.R_Loss_tf, self.R_Loss_Minimize_tf, self.R_Predictions_tf],
                                                          feed_dict={self.R_Inputs_tf: R_batch_train_inputs,
                                                                     self.RanGen_Inputs_tf: np.random.randn(self.R_BatchSize,1),
                                                                     self.R_Labels_tf: R_batch_train_labels,
                                                                     self.R_LearningRate_tf: RegLearnRate})
            
            if i == 0:
                if 'all_reg_weights' in self.AllSumms:
                    R_wb_summ = self.session.run(self.R_param_summaries,
                                                 feed_dict={self.R_Inputs_tf: R_batch_train_inputs,
                                                            self.RanGen_Inputs_tf: np.random.randn(self.R_BatchSize,1),
                                                            self.R_Labels_tf: R_batch_train_labels,
                                                            self.R_LearningRate_tf: RegLearnRate})
                    self.summ_writer.add_summary(R_wb_summ, iteration)
                if 'all_rangen_weights' in self.AllSumms:
                    RanGen_wb_summ = self.session.run(self.Rangen_param_summaries,
                                                      feed_dict={self.R_Inputs_tf: R_batch_train_inputs,
                                                                 self.RanGen_Inputs_tf: np.random.randn(self.R_BatchSize,1),
                                                                 self.R_Labels_tf: R_batch_train_labels,
                                                                 self.R_LearningRate_tf: RegLearnRate})
                    self.summ_writer.add_summary(RanGen_wb_summ, iteration)
                if 'reg_grad' in self.AllSumms:
                    R_gn_summ = self.session.run(self.R_gradnorm_summary,
                                                 feed_dict={self.R_Inputs_tf: R_batch_train_inputs,
                                                            self.RanGen_Inputs_tf: np.random.randn(self.R_BatchSize,1),
                                                            self.R_Labels_tf: R_batch_train_labels,
                                                            self.R_LearningRate_tf: RegLearnRate})
                    self.summ_writer.add_summary(R_gn_summ, iteration)
                
            R_loss_per_iter.append(R_l)
            MSE_per_iter_tr.append(accuracy_MSE(R_train_batch_predictions, R_batch_train_labels))
            MAE_per_iter_tr.append(accuracy_MAE(R_train_batch_predictions, R_batch_train_labels))
            
        # updating Predictions
        AllPredicts = self.PredictAfterTrain(X, RandomIn)
        
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
    
    
    
    
    # =================================== Training of KL-Distance ========================================
    def UpdateKLNet(self, X, Y, RandomIn, iteration, KLLearnRate, nKL_Steps):
        n_samples = np.shape(X)[0]
        
        train_batch_Indeces = batch_index_generator(n_samples, self.R_BatchSize, nKL_Steps)
        
        KL_loss_per_iter = []
        
        for i in range(nKL_Steps):
            KL_batch_train_inputs = X[train_batch_Indeces[:,i],:]
            KL_batch_train_labels = Y[train_batch_Indeces[:,i],:]
            RandomIn = np.random.permutation(RandomIn)
            Ran_batch_train = RandomIn[train_batch_Indeces[:,i],:]

            KL_l,_ = self.session.run([self.KL_Loss_tf, self.KL_Loss_Minimize_tf],
                                      feed_dict={self.R_Inputs_tf: KL_batch_train_inputs,
                                                 self.RanGen_Inputs_tf: np.random.randn(self.R_BatchSize,1),
                                                 self.R_Labels_tf: KL_batch_train_labels,
                                                 self.KL_LearningRate_tf: KLLearnRate})
            if i == 0:
                if 'all_kl_weights' in self.AllSumms:
                    KL_wb_summ = self.session.run(self.KL_param_summaries,
                                                  feed_dict={self.R_Inputs_tf: KL_batch_train_inputs,
                                                             self.RanGen_Inputs_tf: np.random.randn(self.R_BatchSize,1),
                                                             self.R_Labels_tf: KL_batch_train_labels,
                                                             self.KL_LearningRate_tf: KLLearnRate})
                    self.summ_writer.add_summary(KL_wb_summ, iteration)
                if 'kl_grad' in self.AllSumms:
                    KL_gn_summ = self.session.run(self.KL_gradnorm_summary,
                                                  feed_dict={self.R_Inputs_tf: KL_batch_train_inputs,
                                                             self.RanGen_Inputs_tf: np.random.randn(self.R_BatchSize,1),
                                                             self.R_Labels_tf: KL_batch_train_labels,
                                                             self.KL_LearningRate_tf: KLLearnRate})
                    self.summ_writer.add_summary(KL_gn_summ, iteration)
            
            KL_loss_per_iter.append(KL_l)
        
        KL_avg_loss = np.mean(KL_loss_per_iter)
        if 'loss_kl' in self.AllSumms:
            summ_loss_KL = self.session.run(self.KL_summaries_train_tf, feed_dict={self.KL_train_tf: KL_avg_loss})
            self.summ_writer.add_summary(summ_loss_KL, iteration)
        
        return KL_avg_loss
    
    
    
    
    
    
    def TestModel(self, X, Y, RandomIn, iteration):
        n_samples = np.shape(X)[0]
        
        test_batch_Indeces = batch_index_generator(n_samples, self.R_BatchSize, 1)
        
        batch_test_inputs = X[test_batch_Indeces[:,0],:]
        batch_test_labels = Y[test_batch_Indeces[:,0],:]
        Ran_batch_test = RandomIn[test_batch_Indeces[:,0],:]
        
        test_batch_predictions = self.session.run(self.R_Predictions_tf, 
                                                  feed_dict={self.R_Inputs_tf: batch_test_inputs,
                                                            self.RanGen_Inputs_tf: Ran_batch_test})
        
        MSE_per_iter_ts = accuracy_MSE(test_batch_predictions, batch_test_labels)
        MAE_per_iter_ts = accuracy_MAE(test_batch_predictions, batch_test_labels)
        if 'ts_phs' in self.AllSumms:
            summ_test = self.session.run(self.performance_summaries, feed_dict={self.tf_mse_ts: MSE_per_iter_ts,
                                                                                self.tf_mae_ts: MAE_per_iter_ts})
            self.summ_writer.add_summary(summ_test, iteration)
        
        return MSE_per_iter_ts, MAE_per_iter_ts
    
    
    
    
    
    
    
    def GetDist(self, RandomIn):
        n_samples = np.shape(RandomIn)[0]
        NumZeroPad = self.R_BatchSize - n_samples%self.R_BatchSize
        Ran_Ext = np.concatenate([RandomIn, np.zeros(shape=(NumZeroPad, np.shape(RandomIn)[1]))], axis=0)
        n_samples_Ext = np.shape(Ran_Ext)[0]
        Indeces = batch_index_generator(n_samples_Ext, self.R_BatchSize, np.int32(n_samples_Ext/self.R_BatchSize))
        
        
        WholeDists = np.zeros(shape=(n_samples_Ext, 1))
        for i in range(np.int32(n_samples_Ext/self.R_BatchSize)):
            RanIn = Ran_Ext[Indeces[:,i],:]
            Dist = self.session.run([self.RanDist],
                                    feed_dict={self.RanGen_Inputs_tf: RanIn})
            WholeDists[Indeces[:,i],:] = Dist
        
        return WholeDists[:-NumZeroPad, :]
    
    
    
    
    
    
    def PredictAfterTrain(self, X, RandomIn):
        n_samples = np.shape(X)[0]
        NumZeroPad = self.R_BatchSize - n_samples%self.R_BatchSize
        X_ext = np.concatenate([X, np.zeros(shape=(NumZeroPad, np.shape(X)[1]))], axis=0)
        RanInExt = np.concatenate([RandomIn, np.zeros(shape=(NumZeroPad, np.shape(RandomIn)[1]))], axis=0)
        n_samples_Ext = np.shape(X_ext)[0]
        Indeces = batch_index_generator(n_samples_Ext, self.R_BatchSize, np.int32(n_samples_Ext/self.R_BatchSize))
        
        
        WholePreds = np.zeros(shape=(n_samples_Ext, 1))
        for i in range(np.int32(n_samples_Ext/self.R_BatchSize)):
            InFeed = X_ext[Indeces[:,i],:]
            RanIn = RanInExt[Indeces[:,i],:]
            R_preds = self.session.run([self.R_Predictions_tf],
                                       feed_dict={self.R_Inputs_tf: InFeed,
                                                 self.RanGen_Inputs_tf: np.random.randn(self.R_BatchSize,1)})
            WholePreds[Indeces[:,i],:] = R_preds
        
        return WholePreds[:-NumZeroPad, :]
    
    
    
    
    def EndSession(self, CloseSess=1):
              
        if CloseSess:
            self.session.close()
            