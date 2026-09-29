import unittest
from analyze_workspace import cluster_contrast


class AnalysisTests(unittest.TestCase):
    def test_bootstrap_resamples_clusters_not_pseudoreplicated_trials(self):
        rows=[('one',{'a':1,'b':0})]*100+[('two',{'a':0,'b':1})]*100
        result=cluster_contrast(rows,'a','b')
        self.assertEqual(result['clusters'],2)
        self.assertEqual(result['n'],200)
        self.assertEqual(result['difference'],0)
        self.assertEqual(result['ci95'],[-1.,1.])

    def test_missing_and_single_cluster_intervals_are_not_claimed(self):
        self.assertIsNone(cluster_contrast([],'a','b')['ci95'])
        self.assertIsNone(cluster_contrast([('x',{'a':1,'b':0})],'a','b')['ci95'])

    def test_paired_direction_and_count(self):
        rows=[(str(i),{'treated':1,'control':0}) for i in range(20)]
        result=cluster_contrast(rows,'treated','control')
        self.assertEqual(result['difference'],1)
        self.assertEqual(result['ci95'],[1.,1.])
        self.assertEqual(cluster_contrast(rows,'control','treated')['difference'],-1)


if __name__=='__main__':unittest.main()
