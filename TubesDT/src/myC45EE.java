import java.lang.Math;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.classifiers.AbstractClassifier;

import java.util.Enumeration;

public class myC45EE extends AbstractClassifier {
	private myC45EE[] child;
	private Attribute nodeAttribute;
	private double classValue;
	private double errorEstimate = 0;
	
	private double cErrorEstimate = 0.25;
	private double zErrorEstimate = 0.67;
	
	
	@Override
	public void buildClassifier(Instances data) throws Exception {
		data = new Instances(data);
	    data.deleteWithMissingClass();
	    makeTree(data);
	}
	
	private void makeTree(Instances data) throws Exception {
		double[] maxInfoGainData = getMaxInfoGainData(data);
		Instances[] childInstances = null;
		classValue = getMostCommonClass(data);
		if (maxInfoGainData[1] == 0.0) {
			child = null;
		}
		else {
			nodeAttribute = data.attribute((int) maxInfoGainData[0]);
			child = new myC45EE[nodeAttribute.numValues()];
			childInstances = splitInstancesByAttribute(data, nodeAttribute);
			for (int i=0; i<nodeAttribute.numValues(); i++) {
				child[i] = new myC45EE();
				System.out.println("-------["+nodeAttribute+"-"+i+"]");
				if (childInstances[i].numInstances() != 0) {
					child[i].buildClassifier(childInstances[i]);
				}
				
				else {
					System.out.println("-------[empty examples "+nodeAttribute+"-"+i+"]");
					child[i].nodeAttribute = null;
					child[i].classValue = getMostCommonClass(data);
				}
			}
			
			
		}
		
		// C45 pruning
		double N = data.size(); // Number of examples in node
		double f = 0.0; //examples not in node's majority class
		System.out.println("data size:"+data.size());
		if (N != 0) {
			Enumeration<Instance> examplesNodeEnum = data.enumerateInstances();
			while (examplesNodeEnum.hasMoreElements()) {
				Instance inst = (Instance) examplesNodeEnum.nextElement();
				if ((int)inst.classValue() != classValue) {
					f ++;
				}
			}
			f = (double)f/N;
			errorEstimate = getErrorEstimate(f, N);
			
			double errorEstimateChild = 0;
			boolean allChildAreLabel = true;
			if (childInstances != null) {
				for (int i=0; i<nodeAttribute.numValues(); i++){
					if (child[i].child != null) {
						allChildAreLabel = false;
						break;
					} else {
						int NChild = childInstances[i].size();// Number of examples in child node
						errorEstimateChild += (NChild/(double)data.size())*child[i].errorEstimate;
					}
				}
				if (allChildAreLabel) {
					System.out.println("[error estimate]"+errorEstimate+" < [error estimate child]"+errorEstimateChild);
					if (errorEstimate < errorEstimateChild) {
						child = null;
						System.out.println("leaf pruned");
					}
				}
			}
		}
		
		System.out.println("error estimate("+f+","+N+"): "+errorEstimate);
		System.out.println("class Value: "+classValue);
		System.out.println("---------------------------------------------");
	}
	
	private double getErrorEstimate(double f, double N) {
		double temp0 = (f/N) - (Math.pow(f, 2)*1.0/N) + (Math.pow(zErrorEstimate, 2)*1.0/(4.0*Math.pow(N, 2)));
		//System.out.println(temp0);
		double temp1 = zErrorEstimate * Math.sqrt(temp0);
		double temp2 = Math.pow(zErrorEstimate, 2)/(2.0*N);
		double temp3 = temp1 + temp2 + f;
		double divider = 1 + (Math.pow(zErrorEstimate, 2)/N);
		
		//System.out.println("eE("+f+","+N+")"+temp1+"+"+temp2+"+"+f+"/"+divider);
		return temp3/divider;
	}
	
	private double[] getMaxInfoGainData(Instances data) throws Exception {
		double infoGain;
		double maxInfoGain = 0.0;
		double maxInfoGainIdx = 0.0;
		Enumeration<Attribute> attrEnum = data.enumerateAttributes();
		while (attrEnum.hasMoreElements()) {
			Attribute attr = (Attribute) attrEnum.nextElement();
			infoGain = countInfoGain(data, attr);
			if (maxInfoGain < infoGain) {
				maxInfoGain = infoGain;
				maxInfoGainIdx = attr.index();
			}
		}
		double[] maxInfoGainData = {maxInfoGainIdx, maxInfoGain};
		return maxInfoGainData;
	}
	
	private double countInfoGain(Instances data, Attribute attr) throws Exception {
		double infoGain = countEntropy(data);
		
		Instances[] splitInstancesByAttr = new Instances[attr.numValues()];
		splitInstancesByAttr = splitInstancesByAttribute(data, attr);
		
		for (int i=0; i<attr.numValues(); i++) {
			if (splitInstancesByAttr[i].numInstances() != 0) {
				infoGain -= ((double)splitInstancesByAttr[i].numInstances()/(double)data.numInstances()) 
						* countEntropy(splitInstancesByAttr[i]);
			}
		}
		return infoGain;
	}
	
	private double countEntropy(Instances data) throws Exception {
		int countClasses = data.numClasses();
		int countRow = data.numInstances();
		
		double[] classCounts = new double[countClasses];
	    Enumeration<Instance> instEnum = data.enumerateInstances();
	    while (instEnum.hasMoreElements()) {
	    	Instance inst = (Instance) instEnum.nextElement();
	    	classCounts[(int) inst.classValue()] += 1.0;
	    }
	    
	    double entropy = 0;
	    double prob = 0;
	    for (int i=0; i<countClasses; i++) {
	    	prob = classCounts[i] / countRow;
	    	if (prob != 0.0) {
	    		entropy -= prob * Utils.log2(prob);
	    	}
	    }
	    return entropy;
	}
	
	private Instances[] splitInstancesByAttribute(Instances data, Attribute attr) throws Exception {
		Instances[] splitInstancesByAttr = new Instances[attr.numValues()];
		for (int i=0; i<attr.numValues(); i++) {
			splitInstancesByAttr[i] = new Instances(data, data.numInstances());
		}
		
		Enumeration<Instance> instEnum = data.enumerateInstances();
		while (instEnum.hasMoreElements()) {
			Instance inst = (Instance) instEnum.nextElement();
			splitInstancesByAttr[(int) inst.value(attr)].add(inst);
		}
		return splitInstancesByAttr;
	}
	
	private double getMostCommonClass(Instances data) throws Exception {
		int[] countClass = new int[data.numClasses()];
		Enumeration<Instance> instEnum = data.enumerateInstances();
		while (instEnum.hasMoreElements()) {
			Instance inst = (Instance) instEnum.nextElement();
			countClass[(int) inst.classValue()] += 1;
		}
		double mostCommonClass = 0.0;
		int max = 0;
		for (int i=0; i<data.numClasses(); i++) {
			if (countClass[i] > max) {
				max = countClass[i];
				mostCommonClass = (double) i;
			}
		}
		return mostCommonClass;
	}
	
	@Override
	public double classifyInstance(Instance instance) throws Exception {
		if (child == null) {
			return classValue;
		} 
		else {
			return child[(int) instance.value(nodeAttribute)].classifyInstance(instance);
		}
	}
}
