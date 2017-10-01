import java.lang.Math;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;

import java.util.Enumeration;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class myC45 extends AbstractClassifier {
	private treeC45 thisID3;
	
	@Override
	public void buildClassifier(Instances data) throws Exception {
		data = new Instances(data);
	    data.deleteWithMissingClass();
	    data = handleMissingAttributeValue(data);
	    data.randomize(new Random(1));
	    
	    //split data 
	    int trainSize = (int) Math.round(data.numInstances() * 80 / 100);
		int testSize = data.numInstances() - trainSize;
		Instances train = new Instances(data, 0, trainSize);
		Instances test = new Instances(data, trainSize, testSize);
		
		thisID3 = new treeC45();
		thisID3.buildClassifier(train);
		//printTree(thisID3);
	    thisID3 = pruneT(thisID3, test);
	}
	
	private Instances handleMissingAttributeValue(Instances data) throws Exception {
		Instance instance;
		
		Enumeration<Attribute> attrEnum = data.enumerateAttributes();
		while (attrEnum.hasMoreElements()) {
			Attribute attr = (Attribute) attrEnum.nextElement();
	    	if (attr.isNumeric()) {
    			data.sort(attr.index());
	    		Enumeration<Instance> enu = data.enumerateInstances();
	    		int total = 0;
	    		
	    		while (enu.hasMoreElements()) {
	    			instance = enu.nextElement();
	    			
	    			if(instance.isMissing(attr.index())) {
	    				instance.setValue(attr, data.instance(total/2).value(attr.index()));
	    				continue;
	    			}
	    			total++;
	    		}
	    	} else {
    			data.sort(attr.index());
    			Map<Double, Integer> count = new HashMap<>();
	    		Enumeration<Instance> enu = data.enumerateInstances();
	    		
	    		while (enu.hasMoreElements()) {
	    			instance = enu.nextElement();
	    			
	    			if(instance.isMissing(attr.index())) {
	    				Map.Entry<Double, Integer> maxEntry = null;
	    				for (Map.Entry<Double, Integer> entry : count.entrySet()) {
	    				  if (maxEntry == null || entry.getValue() > maxEntry.getValue()) {
	    				    maxEntry = entry;
	    				  }
	    				}
	    				instance.setValue(attr, maxEntry.getKey());
	    				continue;
	    			}
	    			
	    			double attrValue = instance.value(attr.index());
	    			if(count.containsKey(attrValue)) {
		    			count.put(attrValue, count.get(attrValue) + 1);	
	    			} else {
	    				count.put(attrValue, 1);
	    			}
	    		}
	    	}
		}
		
		return data;
	}
	
	private treeC45 pruneT(treeC45 tree, Instances test) throws Exception {
		treeC45 temptree = new treeC45(tree);
		
		if (temptree.getNodeAttribute() != null) {
			if (checkIfAllChildAreLabel(temptree)) {
				Attribute oldattr = temptree.getNodeAttribute();
				temptree.setNodeAttribute(null);
				
				if(!calculateAccuracy(temptree, test)) {
					temptree.setNodeAttribute(oldattr);
				}
			} else {
				for (int i=0; i<(temptree.getNodeAttribute()).numValues(); i++) {
					temptree.setChild(pruneT(temptree.getChild()[i],test), i);
				}
			}			
		}
		return temptree;
	}
	
	private treeC45 pruneTEE(treeC45 tree, Instances test) throws Exception {
		treeC45 temptree = new treeC45(tree);
		if (temptree.getNodeAttribute() != null) {
			if (checkIfAllChildAreLabel(temptree)) {
				System.out.println("-----checking: "+temptree.getNodeAttribute()+"------");
				temptree = compareEstimatedError(temptree);
			} else {
				for (int i=0; i<(temptree.getNodeAttribute()).numValues(); i++) {
					temptree.setChild(pruneTEE(temptree.getChild()[i],test), i);
				}
				if (checkIfAllChildAreLabel(temptree)) {
					temptree = compareEstimatedError(temptree);
				}
			}
		}
		return temptree;
	}
	
	private treeC45 compareEstimatedError(treeC45 tree) {
		treeC45 temptree = new treeC45(tree);
		Attribute oldattr = temptree.getNodeAttribute();
		System.out.println(temptree.getExamplesNode());
		System.out.println(temptree.getClassValue());
		double N = temptree.getExamplesNode().size();
		double f = 0.0; //examples not in node's majority class
		double errorEstimateChild = 0.0;
		for (int i=0; i<oldattr.numValues(); i++){
				int NChild = temptree.getChild()[i].getExamplesNode().size();// Number of examples in child node
				errorEstimateChild += (NChild/(double)temptree.getExamplesNode().size())*temptree.getChild()[i].getErrorEstimate();
		}
		
		System.out.println("[error estimate]"+temptree.getErrorEstimate()+" < [error estimate child]"+errorEstimateChild);
		if (temptree.getErrorEstimate() < errorEstimateChild) {
			temptree.setNodeAttribute(null);
			System.out.println("leaf pruned");
		}
		return temptree;
	}
	
	private boolean calculateAccuracy(treeC45 prunedTree, Instances test) throws Exception{
		//get complete tree
		treeC45 aftertree = new treeC45(prunedTree);
		while(aftertree.getParent() != null) {
			treeC45 parenttree = new treeC45(aftertree.getParent());
			parenttree.setChild(aftertree, aftertree.getIndex());
			aftertree = parenttree;
		}
		
		Evaluation eval_before = new Evaluation(test);
		Evaluation eval_after = new Evaluation(test);
		
		eval_before.evaluateModel(thisID3, test);
		eval_after.evaluateModel(aftertree, test);
		
		double before_accuracy = eval_before.pctCorrect();
		double after_accuracy = eval_after.pctCorrect();

//		System.out.println();
//		System.out.println("[after:"+after_accuracy+"] >= [before:"+before_accuracy+"]");
//		System.out.println();
		
		if (after_accuracy > before_accuracy) {
			//System.out.println("pruned");
			thisID3 = aftertree;
			return true;
		}
		return false;
	}
	
	private void printTree(treeC45 tree) {
		if (tree.getNodeAttribute() == null) {
			System.out.println("["+tree.getClassValue()+"]");
		}
		else {
			System.out.println("["+tree.getNodeAttribute()+"(cv:"+tree.getClassValue()+")]");
			for (int i=0; i<(tree.getNodeAttribute()).numValues(); i++) {
				System.out.print("["+tree.getNodeAttribute()+"-"+i+"/"+tree.getIndex()+"]");
				printTree(tree.getChild()[i]);
			}
		}
	}
	
	private boolean checkIfAllChildAreLabel(treeC45 tree) {
		for (int i=0; i<(tree.getNodeAttribute()).numValues(); i++){
			if ((tree.getChild())[i].getNodeAttribute() != null) {
				return false;
			}
		}
		return true;
	}
	
	@Override
	public double classifyInstance(Instance instance) throws Exception {
		return thisID3.classifyInstance(instance);
	}
}