import java.lang.Math;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;

import java.util.Enumeration;
import java.util.Random;

public class myC45 extends AbstractClassifier {
	private treeC45 thisID3;
	
	@Override
	public void buildClassifier(Instances data) throws Exception {
		data = new Instances(data);
	    data.deleteWithMissingClass();
	    data.randomize(new Random(1));
	    //split data 
	    int trainSize = (int) Math.round(data.numInstances() * 80 / 100);
		int testSize = data.numInstances() - trainSize;
		Instances train = new Instances(data, 0, trainSize);
		Instances test = new Instances(data, trainSize, testSize);
		
		thisID3 = new treeC45();
		thisID3.buildClassifier(train);
	    thisID3 = pruneT(thisID3, test);
	}
	
	private treeC45 pruneT(treeC45 tree, Instances test) throws Exception {
		treeC45 temptree = new treeC45(tree);
		
		if (temptree.getNodeAttribute() != null) {
			if (checkIfAllChildAreLabel(temptree)) {
				//System.out.println("-------atribut pruned:"+temptree.getNodeAttribute());
				
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