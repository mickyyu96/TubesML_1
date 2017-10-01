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
	private treeC45 thisID3;
	private ArrayList<ruleC45> rules;
	
	@Override
	public void buildClassifier(Instances data) throws Exception {
		data = new Instances(data);
	    data.deleteWithMissingClass();
	    data.randomize(new Random(1));
	    //split data 
		rules = new ArrayList<ruleC45>();
		thisID3 = new treeC45();
		thisID3.buildClassifier(data);
		//printTree(thisID3);
		System.out.println();
	    translateToRules(thisID3);
	    pruneRules(data);
	}
	
	private void translateToRules(treeC45 tree) throws Exception {
		treeC45 temptree = new treeC45(tree);
		if (temptree.getNodeAttribute() == null) {
			addRules(temptree);
		} else {
			for (int i=0; i<(temptree.getNodeAttribute()).numValues(); i++) {
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
			newRule.addPrecond(parenttree.getNodeAttribute(), beforetree.getIndex());
			beforetree = parenttree;
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
		printRules();
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
			System.out.println("["+tree.getNodeAttribute()+"(cv:"+tree.getClassValue()+")]");
			for (int i=0; i<(tree.getNodeAttribute()).numValues(); i++) {
				System.out.print("["+tree.getNodeAttribute()+"-"+i+"/"+tree.getIndex()+"]");
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
		return thisID3.getClassValue();
	}
}