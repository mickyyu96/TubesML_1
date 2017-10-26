import java.lang.Math;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import java.util.*;

import java.util.Enumeration;
import java.util.Random;

public class myC45RP extends AbstractClassifier {
	private treeC45 thisTree = new treeC45();
	private ArrayList<ruleC45> rules;
	private int method = 0;
	@Override
	public void buildClassifier(Instances data) throws Exception {
		data = new Instances(data);
	    data.deleteWithMissingClass();
	    data.randomize(new Random(1));
	    
		rules = new ArrayList<ruleC45>();
		thisTree.attributeSelectionMethod(method);
		thisTree.buildClassifier(data);
	    translateToRules(thisTree);
	    //System.out.println("--------------start pruning-------------");
	    //printRules();
	    pruneRules(data);
	    //System.out.println("===========================after pruning:");
	    //printRules();
	   
	}
	
	public void setMethod(int x) {
		method = x;
	}
	
	private void translateToRules(treeC45 tree) throws Exception {
		treeC45 temptree = new treeC45(tree);
		if (temptree.getNodeAttribute() == null) {
			addRules(temptree);
		} else {
			int N = temptree.getNodeAttribute().numValues();
			if (temptree.getNodeAttribute().isNumeric()) {
				N = 2;
			}
			for (int i=0; i< N; i++) {
				translateToRules(temptree.getChild()[i]);
			}
		}
	}
	
	private void addRules(treeC45 tree) {
		treeC45 beforetree = new treeC45(tree);
		ruleC45 newRule = new ruleC45();
		newRule.addClassValue(tree.getClassValue());
		while(beforetree.getParent() != null) {
			treeC45 parenttree = new treeC45(beforetree.getParent());
			if (parenttree.getNodeAttribute().isNumeric()) {
				//newRule.addSplitPrecond(i, parenttree.getSplitPoint());
				newRule.addPrecond(parenttree.getNodeAttribute(), beforetree.getIndex(), parenttree.getSplitPoint());
			} else {
				newRule.addPrecond(parenttree.getNodeAttribute(), beforetree.getIndex(), -1.0);
			}
			beforetree = parenttree;
			if(beforetree.getParent() != null) {
				//System.out.println(beforetree.getParent().getNodeAttribute());
			}
		}
		rules.add(newRule);
	}
	
	private void pruneRules(Instances test) {
		for (ruleC45 rule: rules) {;
			rule.prune(test);
		}
		ArrayList<ruleC45> orderedRules = new ArrayList<ruleC45>();
		while(!rules.isEmpty()) {
			double maxacc = rules.get(0).getAccuracy();
			int maxi = 0;
			for(int i=1;i<rules.size();i++) {
				if(rules.get(i).getAccuracy()>maxacc) {
					maxacc = rules.get(i).getAccuracy();
					maxi = i;
				}
			}
			ruleC45 temp = rules.remove(maxi);
			orderedRules.add(temp);
		}
		rules = new ArrayList<ruleC45>(orderedRules);
	}
	
	private void printRules() {
		for (int i = 0; i < rules.size(); i++) {
			rules.get(i).printRule();
		}
	}
	
	private void printTree(treeC45 tree) {
		if (tree.getNodeAttribute() == null) {
			System.out.println("["+tree.getClassValue()+"]");
		}
		else {
			System.out.println("["+tree.getNodeAttribute()+"(cv:"+tree.getClassValue()+")]    Split point: " + tree.splitPoint);
			for (int i=0; i<(tree.getChild()).length; i++) {
				System.out.print("["+tree.getNodeAttribute()+"-"+i+"/"+tree.getIndex()+"]" );
				printTree(tree.getChild()[i]);
			}
		}
	}
	
	@Override
	public double classifyInstance(Instance instance) throws Exception {
		for (int i = 0; i < rules.size(); i++) {
			if(rules.get(i).classify(instance)) {
				return rules.get(i).getClassValue();
			} 
		}
		return thisTree.getClassValue();
	}
}