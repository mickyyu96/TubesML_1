import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.classifiers.AbstractClassifier;

import java.util.Enumeration;

public class myC45 extends AbstractClassifier {
	private myC45[] child;
	private Attribute nodeAttribute;
	private double classValue;
	
	@Override
	public void buildClassifier(Instances data) throws Exception {
		data = new Instances(data);
	    data.deleteWithMissingClass();
	    
	    for (int i = 0; i < data.numAttributes(); i++) {
	    	if (data.attribute(i).isNumeric()) {
	    		data.sort(data.attribute(i));
	    		handleNumericAttribute(data);
	    	}
	    }
	}
	
	private void handleNumericAttribute(Instances data) {
		
	}
	
	private void makeTree(Instances data) throws Exception {
		double[] maxInfoGainData = getMaxInfoGainData(data);
		if (maxInfoGainData[1] == 0.0) {
			nodeAttribute = null;
			classValue = getMostCommonClass(data);
		}
		else {
			nodeAttribute = data.attribute((int) maxInfoGainData[0]);
			child = new myC45[nodeAttribute.numValues()];
			Instances[] childInstances = splitInstancesByAttribute(data, nodeAttribute);
			for (int i=0; i<nodeAttribute.numValues(); i++) {
				child[i] = new myC45();
				if (childInstances[i].numInstances() != 0) {
					child[i].buildClassifier(childInstances[i]);
				}
				else {
					child[i].nodeAttribute = null;
					child[i].classValue = getMostCommonClass(data);
				}
			}
		}
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
		if (nodeAttribute == null) {
			return classValue;
		} 
		else {
			return child[(int) instance.value(nodeAttribute)].classifyInstance(instance);
		}
	}
}
