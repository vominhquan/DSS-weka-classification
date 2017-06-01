/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package dss;

import java.awt.Font;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.JTextArea;
import javax.swing.SwingUtilities;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.Vote;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import java.util.concurrent.ThreadLocalRandom;


/**
 *
 * @author VMQ
 */

public class DSS {

    /**
     * @param args the command line arguments
     */
    
    public static JFrame mainFrame;
    public static JPanel input;
    public static JPanel result;
    public static JTable jt;
    public static File trainFile;
    public static Logistic logis;
    public static SMO svm ;
    public static J48 dt;
    public static RandomForest rf ;
    public static AdaBoostM1 bt ;
    public static Vote vote ;
    public static MultilayerPerceptron mlp;
    public static Evaluation eval;
    public static Evaluation eval2;
    public static Evaluation eval3;
    public static Evaluation eval4;
    public static Evaluation eval5;
    public static Evaluation eval6;
    public static Evaluation eval7;
    public static JTextArea as;
    public static DataSource trainsource;
    public static JTextArea taVote;
    
    
    public static void savemodel() throws Exception{
                // save model
//        weka.core.SerializationHelper.write("trained model\\AdaBoostM1.model", bt);
//        weka.core.SerializationHelper.write("trained model\\RandomForest.model", rf);
//        weka.core.SerializationHelper.write("trained model\\J48.model", dt);
//        weka.core.SerializationHelper.write("trained model\\MultilayerPerceptron.model", mlp);
//        weka.core.SerializationHelper.write("trained model\\Logistic.model", logis);
//        weka.core.SerializationHelper.write("trained model\\SMO.model", svm);
        weka.core.SerializationHelper.write("trained model\\Vote.model", vote);
    }
    
    public static void trainfile(String trainpath) throws Exception{
        trainsource = new DataSource(trainpath);
        Instances datatrain = trainsource.getDataSet();
        if (datatrain.classIndex() == -1)
            datatrain.setClassIndex(datatrain.numAttributes() - 1);
        
        //build model
        logis = new Logistic();
        logis.buildClassifier(datatrain);
        mlp = new MultilayerPerceptron();
        mlp.buildClassifier(datatrain);
        dt = new J48();
        dt.buildClassifier(datatrain);
        rf = new RandomForest();
        rf.buildClassifier(datatrain);
        bt = new AdaBoostM1();
        bt.buildClassifier(datatrain);
        svm = new SMO();
        svm.buildClassifier(datatrain);

        Classifier[] classifiers = {logis,mlp,dt,rf,bt,svm};
        vote.setClassifiers(classifiers);
        vote.buildClassifier(datatrain);
        weka.core.SerializationHelper.write("trained model\\Vote.model", vote);
        
        // evaluate
        DataSource sourcetest = new DataSource("data\\testdata.arff");
           Instances datatest = sourcetest.getDataSet();
           if (datatest.classIndex() == -1)
               datatest.setClassIndex(datatest.numAttributes() - 1);
            
        eval = new Evaluation(datatest);
        eval.evaluateModel(logis, datatest);
        eval2 = new Evaluation(datatest);
        eval2.evaluateModel(mlp, datatest);
        eval3 = new Evaluation(datatest);
        eval3.evaluateModel(dt, datatest);
        eval4 = new Evaluation(datatest);
        eval4.evaluateModel(rf, datatest);
        eval5 = new Evaluation(datatest);
        eval5.evaluateModel(bt, datatest);
        eval6 = new Evaluation(datatest);
        eval6.evaluateModel(svm, datatest);
        eval7 = new Evaluation(datatest);
        eval7.evaluateModel(vote, datatest);
        as.setText("Logistic Regression: " + String.valueOf(eval.pctCorrect())
                + "%\nNeural Network: "+ String.valueOf(eval.pctCorrect())
                + "%\nDecision Tree: "+ String.valueOf(eval.pctCorrect())
                + "%\nSupport Vector Machine: "+ String.valueOf(eval.pctCorrect())
                + "%\nRandom Forest: "+ String.valueOf(eval.pctCorrect())
                + "%\nBoosted Tree: "+ String.valueOf(eval.pctCorrect()));
    }
    
    public static void predictfile(String predictpath) throws Exception{
        DataSource predictsource = new DataSource(predictpath);
        Instances datapredict = predictsource.getDataSet();
        if (datapredict.classIndex() == -1)
            datapredict.setClassIndex(datapredict.numAttributes() - 1);
        ////////////////////////////// prepare table
        //prepare data
        int numKhong =0;
        for (int j=0;j<datapredict.numInstances();j++){
            Instance newInst = datapredict.instance(j);
            double preNB = vote.classifyInstance(newInst);
            String preString = datapredict.classAttribute().value((int) preNB);
            if(preString.compareTo("Khong")==0){
                numKhong++;
            }
        }
        int index =0;
        Object[][] tabledata = new Object[numKhong][30];
        for (int j=0;j<datapredict.numInstances();j++){

            Instance newInst = datapredict.instance(j);
            double preNB = vote.classifyInstance(newInst);
            String preString = datapredict.classAttribute().value((int) preNB);
//            double actualClass = datatest.instance(j).classValue();
//            String actual = datatest.classAttribute().value((int) actualClass);
            //System.out.print(j+", "+actual+", "+preString+"\n");
            if(preString.compareTo("Khong")==0){
                ArrayList<Object> str = new ArrayList();            
                Object[] obj =  datapredict.instance(j).toString().split(",");
                str.addAll(Arrays.asList(obj));
                str.add(preString);
                tabledata[index] = str.toArray();
                index++;
            }
        }
        //prepare header
        Object[] columns = new Object[30];
        for (int j=0;j<datapredict.numAttributes();j++){
            Object obj = datapredict.attribute(j).toString().split(" ")[1];
            columns[j] = obj; 
        }
        columns[29] = "Predict";
        ////////////////////// end prepare table
        jt = new JTable(tabledata,columns);
        
        JScrollPane scrollPane = new JScrollPane(jt);
        result.removeAll();
        result.add(scrollPane);
        SwingUtilities.updateComponentTreeUI(result);
//        mainFrame.setExtendedState(JFrame.MAXIMIZED_BOTH); 
    }
    
    public static void balancedata() throws Exception{
        Instances datatrain = trainsource.getDataSet();
        if (datatrain.classIndex() == -1)
            datatrain.setClassIndex(datatrain.numAttributes() - 1);
        
        Instances arr_khong = new Instances(datatrain,0);
        Instances arr_co = new Instances(datatrain,0);
        
        for (int j=0;j<datatrain.numInstances();j++){            
            double actualClass = datatrain.instance(j).classValue();
            String actual = datatrain.classAttribute().value((int) actualClass);
            if(actual.compareTo("Co")==0){
                arr_co.add(datatrain.instance(j));
            }
            else{
                arr_khong.add(datatrain.instance(j));
            }
        }
        System.out.println(arr_khong.numInstances());
        System.out.println(arr_co.numInstances());
        System.out.println(datatrain.numInstances());
        
        
        int num =arr_co.numInstances()-arr_khong.numInstances();

        if (arr_co.numInstances()>arr_khong.numInstances())
            for (int j=0;j<num;j++){         
                int randomNum = ThreadLocalRandom.current().nextInt(0, arr_co.numInstances());
                System.out.println(j+" "+randomNum+" "+arr_co.numInstances());
                arr_co.delete(randomNum);
            }
        for (int j=0;j<arr_khong.numInstances();j++){         
              arr_co.add(arr_khong.instance(j));  
        }   

        //build model
        logis = new Logistic();
        logis.buildClassifier(datatrain);
        mlp = new MultilayerPerceptron();
        mlp.buildClassifier(datatrain);
        dt = new J48();
        dt.buildClassifier(datatrain);
        rf = new RandomForest();
        rf.buildClassifier(datatrain);
        bt = new AdaBoostM1();
        bt.buildClassifier(datatrain);
        svm = new SMO();
        svm.buildClassifier(datatrain);

        Classifier[] classifiers = {logis,mlp,dt,rf,bt,svm};
        vote.setClassifiers(classifiers);
        vote.buildClassifier(datatrain);
        weka.core.SerializationHelper.write("trained model\\Vote.model", vote);
        
        // evaluate
        DataSource sourcetest = new DataSource("data\\testdata.arff");
           Instances datatest = sourcetest.getDataSet();
           if (datatest.classIndex() == -1)
               datatest.setClassIndex(datatest.numAttributes() - 1);
            
        eval = new Evaluation(datatest);
        eval.evaluateModel(logis, datatest);
        eval2 = new Evaluation(datatest);
        eval2.evaluateModel(mlp, datatest);
        eval3 = new Evaluation(datatest);
        eval3.evaluateModel(dt, datatest);
        eval4 = new Evaluation(datatest);
        eval4.evaluateModel(rf, datatest);
        eval5 = new Evaluation(datatest);
        eval5.evaluateModel(bt, datatest);
        eval6 = new Evaluation(datatest);
        eval6.evaluateModel(svm, datatest);
        eval7 = new Evaluation(datatest);
        eval7.evaluateModel(vote, datatest);
        as.setText("Logistic Regression: " + String.valueOf(eval.pctCorrect())
                + "%\nNeural Network: "+ String.valueOf(eval.pctCorrect())
                + "%\nDecision Tree: "+ String.valueOf(eval.pctCorrect())
                + "%\nSupport Vector Machine: "+ String.valueOf(eval.pctCorrect())
                + "%\nRandom Forest: "+ String.valueOf(eval.pctCorrect())
                + "%\nBoosted Tree: "+ String.valueOf(eval.pctCorrect()));
    }
    
    public static void main(String[] args) throws Exception {
//        trainsource = new DataSource("data\\original_data.arff");
//        Instances datatrain = trainsource.getDataSet();
//        if (datatrain.classIndex() == -1)
//            datatrain.setClassIndex(datatrain.numAttributes() - 1);
//        balancedata();
        
        /////////////////// read model
//        bt2 = (AdaBoostM1) weka.core.SerializationHelper.read("trained model\\AdaBoostM1.model");
        vote = (Vote) weka.core.SerializationHelper.read("trained model\\Vote.model");
        
        ////////////////// button event
        JButton predict_file = new JButton("predict file");
        predict_file.addActionListener((ActionEvent e) -> {
            JFileChooser fileChooser = new JFileChooser("data");
            int resultf = fileChooser.showOpenDialog(new JFrame());
            if (resultf == JFileChooser.APPROVE_OPTION) {
                 String predictpath =fileChooser.getSelectedFile().getAbsolutePath();
                try {
                    predictfile(predictpath);
                } catch (Exception ex) {
                    Logger.getLogger(DSS.class.getName()).log(Level.SEVERE, null, ex);
                }
            }
        });
        JButton train_file = new JButton("train file");
        train_file.addActionListener((ActionEvent e) -> {
            JFileChooser fileChooser = new JFileChooser("data");
            int resultf = fileChooser.showOpenDialog(new JFrame());
            if (resultf == JFileChooser.APPROVE_OPTION) {
                 String trainpath =fileChooser.getSelectedFile().getAbsolutePath();
                try {
                    trainfile(trainpath);
                } catch (Exception ex) {
                    Logger.getLogger(DSS.class.getName()).log(Level.SEVERE, null, ex);
                }
            }
        });
        
        JButton savemodel = new JButton("save model");
        savemodel.addActionListener((ActionEvent e) -> {
            try {
                savemodel();
            } catch (Exception ex) {
                Logger.getLogger(DSS.class.getName()).log(Level.SEVERE, null, ex);
            }
        });
        JButton balance = new JButton("balance data");
        balance.addActionListener((ActionEvent e) -> {
            try {
                balancedata();
            } catch (Exception ex) {
                Logger.getLogger(DSS.class.getName()).log(Level.SEVERE, null, ex);
            }
        });
        
//        trainfile("data\\balanced1.arff");
        
        ////////////////////////////////////////////// GUI
        //////////////////////////input
        JPanel leftinput = new JPanel();
        JPanel rightinput = new JPanel();
        leftinput.add(predict_file);
        rightinput.add(train_file);
        rightinput.add(savemodel);
        rightinput.add(balance);
        input = new JPanel(new GridLayout(1,2,10,10));
        input.add(leftinput);
        input.add(rightinput);

        //////////////////////////accu
        DataSource sourcetest = new DataSource("data\\testdata.arff");
            Instances datatest = sourcetest.getDataSet();
            if (datatest.classIndex() == -1)
               datatest.setClassIndex(datatest.numAttributes() - 1);
        eval7 = new Evaluation(datatest);
        eval7.evaluateModel(vote, datatest);
        
        taVote = new JTextArea();
        taVote.setEditable(false);
        taVote.append(eval7.toClassDetailsString()+"\n");
        taVote.append(eval7.toMatrixString());
        JScrollPane tempscroll = new JScrollPane(taVote);
        
        JPanel accu = new JPanel(new GridLayout(1,2,10,10));
        JPanel leftaccu = new JPanel(new GridLayout(2,1,10,10));
        JPanel rightaccu = new JPanel(new GridLayout(1,1,10,10));
        leftaccu.add(new JLabel("Voting"));
        leftaccu.add(tempscroll);
        accu.add(leftaccu);
        accu.add(rightaccu);
        as = new JTextArea();
        Font font = as.getFont();
        as.setFont( font.deriveFont(26f) );
        as.setEditable(false);
        
        as.append("Logistic Regression: 79.3%\n"
                + "Neural Network: 78.8%\n"
                + "Decision Tree: 80.1%\n"
                + "Support Vector Machine: 79.7%\n"
                + "Random Forest: 79.2%\n"
                + "Boosted Tree: 80.1%");
        rightaccu.add(as);

        //////////////////////main
        mainFrame = new JFrame("Phát hiện sinh viên có nguy cơ nghỉ học");
        mainFrame.setLayout(new GridLayout(3,1,10,10));
        mainFrame.setExtendedState(JFrame.MAXIMIZED_BOTH); 
        mainFrame.setSize(2000, 3000);
        mainFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        
        result = new JPanel(new GridLayout(1,1,10,10));
        
        mainFrame.add(input);
        mainFrame.add(result);
        mainFrame.add(accu);
        mainFrame.show();
        
    }
    
}
