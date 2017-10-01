import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.classifiers.AbstractClassifier;

import java.util.Enumeration;

public class treeC45 extends AbstractClassifier {
	private treeC45 parent;
	int indexattr = -1;
	private double splitPoint = Double.MAX_VALUE; // for numeric attribute
	private treeC45[] child;
	private Attribute nodeAttribute;
	private double classValue;
	
	public treeC45() {}
	public treeC45(treeC45 tree) {
		parent = tree.parent;
		indexattr = tree.indexattr;
		splitPoint = tree.splitPoint;
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
			if(nodeAttribute.isNumeric()) {
				child = new treeC45[2];

	    		data.sort(nodeAttribute);
	    		splitC45 childInstances = new splitC45();
	    		childInstances.handleNumericAttribute(nodeAttribute.index(), data);
				
		    		if(childInstances.isSplit()) {
		    			splitPoint = childInstances.splitPoint();
		    			child[0] = new treeC45();
		    			child[1] = new treeC45();
		    			child[0].indexattr = 0;
		    			child[1].indexattr = 1;
		    			child[0].parent = this;
		    			child[1].parent = this;
					child[0].buildClassifier(childInstances.leftInstances());
					child[1].buildClassifier(childInstances.rightInstances());
		    		} else {
		    			child[0] = new treeC45();
					child[0].nodeAttribute = null;
					child[0].classValue = getMostCommonClass(data);
		    		}
	    		
			} else {
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
						child[i].indexattr = i;
						child[i].classValue = getMostCommonClass(data);
					}
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
		    	if (attr.isNumeric()) {
		    		data.sort(attr.index());
		    		splitC45 numAttr = new splitC45();
		    		numAttr.handleNumericAttribute(attr.index(), data);
		    		infoGain = numAttr.infoGain();
		    	} else {
		    		infoGain = countInfoGain(data, attr);
		    	}
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
//		System.out.println();
//		System.out.println("----calculate Gain Ratio----");
//		System.out.println("avg gain:"+avgGain);
		Enumeration<Attribute> attrEnum = data.enumerateAttributes();
		while (attrEnum.hasMoreElements()) {
			Attribute attr = (Attribute) attrEnum.nextElement();
		 	if (attr.isNumeric()) {
		    		data.sort(attr.index());
		    		splitC45 numAttr = new splitC45();
		    		numAttr.handleNumericAttribute(attr.index(), data);
		    		gain = numAttr.infoGain();
		    		double prob0 = numAttr.getPerBag()[0]/(double)data.size();
		    		double prob1 = numAttr.getPerBag()[1]/(double)data.size();
		    		splitInformation = - (prob0 * Utils.log2(prob0)) - (prob1 * Utils.log2(prob1));
		    	} else {
		    		gain = countInfoGain(data, attr);
		    		splitInformation = countSplitInformation(data, attr);
		    	}
//			System.out.println("===="+attr);
			
//			System.out.println("gain:"+gain);
//			System.out.println("split information:"+countSplitInformation(data, attr));
//			System.out.println("gain ratio:"+((double)gain/countSplitInformation(data, attr)));
			if (gain >= avgGain) {
				double gainRatio = gain/splitInformation;	
				if (gainRatio > maxGainRatio) {
					maxGainRatio = gainRatio;
					maxGainRatioIdx = attr.index();
				}
			}
		}

		//System.out.println("<"+maxGainRatioIdx+", "+maxGainRatio+">");
		double[] maxGainRatioData = {maxGainRatioIdx, maxGainRatio};
		return maxGainRatioData;
	}
	
	private double getAvgGainData(Instances data) throws Exception {
		double infoGain;
		double totalInfoGain = 0.0;
		Enumeration<Attribute> attrEnum = data.enumerateAttributes();
		//System.out.println(">>>>getAvgGainData");
		while (attrEnum.hasMoreElements()) {
			Attribute attr = (Attribute) attrEnum.nextElement();
			if (attr.isNumeric()) {
		    		data.sort(attr.index());
		    		splitC45 numAttr = new splitC45();
		    		numAttr.handleNumericAttribute(attr.index(), data);
		    		infoGain = numAttr.infoGain();
			} else {
				infoGain = countInfoGain(data, attr);
			}
			//System.out.println("gain "+attr+" : "+infoGain);
			totalInfoGain += infoGain;
			
		}
		double avgInfoGainData = totalInfoGain/((double)data.numAttributes()-1);
		System.out.println(totalInfoGain+" / "+((double)data.numAttributes()-1)+"="+avgInfoGainData);
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
				prob = splitInstancesByAttr[i].numInstances() / (double)data.size();
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
			if (nodeAttribute.isNumeric()) {
				if((double) instance.value(nodeAttribute) <= splitPoint) {
					child[0].classifyInstance(instance);
				} else {
					child[1].classifyInstance(instance);
				}
			}
			return child[(int) instance.value(nodeAttribute)].classifyInstance(instance);
		}
	}
}
