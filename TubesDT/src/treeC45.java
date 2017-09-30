import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.classifiers.AbstractClassifier;

import java.util.Enumeration;

public class treeC45 extends AbstractClassifier {
	private treeC45 parent;
	int indexattr = -1;
	private treeC45[] child;
	private Attribute nodeAttribute;
	private double classValue;
	
	public treeC45() {}
	public treeC45(treeC45 tree) {
		parent = tree.parent;
		indexattr = tree.indexattr;
		if (tree.nodeAttribute != null) {
			child = new treeC45[tree.nodeAttribute.numValues()];
			for (int i = 0; i<tree.nodeAttribute.numValues(); i++) {
				child[i] = new treeC45(tree.child[i]);
			}
		}
		nodeAttribute = tree.nodeAttribute; 
		classValue = tree.classValue; 
	}
	
	@Override
	public void buildClassifier(Instances data) throws Exception {
		data = new Instances(data);
	    data.deleteWithMissingClass();
	    makeTree(data);
	}
	
	private void makeTree(Instances data) throws Exception {
		double[] maxInfoGainData = getMaxGainRatioData(data);
		classValue = getMostCommonClass(data);
		if (maxInfoGainData[1] == 0.0) {
			nodeAttribute = null;
		}
		else {
			nodeAttribute = data.attribute((int) maxInfoGainData[0]);
			child = new treeC45[nodeAttribute.numValues()];
			Instances[] childInstances = splitInstancesByAttribute(data, nodeAttribute);
			for (int i=0; i<nodeAttribute.numValues(); i++) {
				child[i] = new treeC45();
				child[i].parent = this;
				child[i].indexattr = i;
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
	
	private double[] getMaxGainRatioData(Instances data) throws Exception {
		double gain;
		double splitInformation;
		double maxGainRatio = 0.0;
		double maxGainRatioIdx = 0.0;
		double avgGain = getAvgGainData(data);
		Enumeration<Attribute> attrEnum = data.enumerateAttributes();
		while (attrEnum.hasMoreElements()) {
			Attribute attr = (Attribute) attrEnum.nextElement();
			gain = countInfoGain(data, attr);
			if (gain >= avgGain) {
				splitInformation = countSplitInformation(data, attr);
				double gainRatio = gain/splitInformation;
				if (gainRatio > maxGainRatio) {
					maxGainRatio = gainRatio;
					maxGainRatioIdx = attr.index();
				}
			}
		}
		double[] maxGainRatioData = {maxGainRatioIdx, maxGainRatio};
		return maxGainRatioData;
	}
	
	private double getAvgGainData(Instances data) throws Exception {
		double infoGain;
		double totalInfoGain = 0.0;
		Enumeration<Attribute> attrEnum = data.enumerateAttributes();
		while (attrEnum.hasMoreElements()) {
			Attribute attr = (Attribute) attrEnum.nextElement();
			infoGain = countInfoGain(data, attr);
			totalInfoGain += infoGain;
		}
		double avgInfoGainData = totalInfoGain/data.numAttributes();
		return avgInfoGainData;
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
	
	private double countSplitInformation(Instances data, Attribute attr) throws Exception{
		double entropy = 0;
		double prob = 0;
		
		Instances[] splitInstancesByAttr = new Instances[attr.numValues()];
		splitInstancesByAttr = splitInstancesByAttribute(data, attr);
		
		for (int i=0; i<attr.numValues(); i++) {
			if (splitInstancesByAttr[i].numInstances() != 0) {
				
				prob = splitInstancesByAttr[i].numInstances() / data.size();
			    	if (prob != 0.0) {
			    		entropy -= prob * Utils.log2(prob);
			    	}
			}
		}
		return entropy;
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
	
	public treeC45[] getChild() {
		return child;
	}
	
	public Attribute getNodeAttribute() {
		return nodeAttribute;
	}
	
	public void setNodeAttribute(Attribute attr) {
		nodeAttribute = attr;
	}
	
	public double getClassValue() {
		return classValue;
	}
	
	public void setChild(treeC45 tree, int i) {
		child[i] = tree;
	}
	
	public treeC45 getParent() {
		return parent;
	}
	
	public int getIndex() {
		return indexattr;
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
