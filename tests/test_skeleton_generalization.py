"""Run from repo root: python -m unittest discover -s tests -p test_skeleton_generalization.py"""
import random
import unittest

import torch
from model.skeleton import H36M, MPI, SkeletonSpec, sample_layout, eval_layout, make_layout, TRAIN_DROP, TRAIN_EDGES
from model.SkeletonAGFormer import SkeletonAGFormer
from utils.skeleton_experiment import project_to_fixed


class SkeletonGeneralizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def small_model(self, **kw):
        opts = dict(n_layers=2, dim_feat=16, dim_rep=32, num_heads=4, n_frames=5, neighbour_num=2)
        opts.update(kw)
        return SkeletonAGFormer(**opts)

    def test_reindex_and_graph(self):
        order = list(reversed(range(17)))
        perm = H36M.reindex(order)
        self.assertEqual(perm.root, 16)
        a = H36M.tensors("cpu", torch.float32)
        b = perm.tensors("cpu", torch.float32)
        torch.testing.assert_close(b[0], a[0][order])
        for v,w in zip(a[1:],b[1:]):
            torch.testing.assert_close(w, v[order][:,order])
        self.assertEqual(MPI.root, 14)
        self.assertEqual(a[1][1,0].item(),1)
        self.assertEqual(a[1][0,1].item(),2)
        self.assertEqual(a[2][0,3].item(),3)
        self.assertTrue(torch.isfinite(a[3]).all())

    def test_layout_holdout_and_baseline(self):
        x = torch.randn(2,5,17,3)
        sparse = eval_layout(H36M,"holdout_sparse")
        dense = eval_layout(H36M,"holdout_dense")
        mixed = eval_layout(H36M,"holdout_mixed")
        self.assertEqual([len(l.spec.names) for l in (sparse,dense,mixed)],[13,21,17])
        self.assertNotEqual(mixed.spec.names,H36M.names)
        for layout in (sparse,dense,mixed):
            observed = layout.apply(x)
            projected = project_to_fixed(observed,layout.spec,H36M)
            self.assertEqual(projected.shape,x.shape)
            for i,name in enumerate(H36M.names):
                if name in layout.spec.names:
                    torch.testing.assert_close(projected[:,:,i],x[:,:,i])
        for i,row in enumerate(dense.rows):
            a,b,alpha = row
            torch.testing.assert_close(dense.apply(x)[:,:,i,:2],
                                       torch.lerp(x[:,:,a,:2],x[:,:,b,:2],alpha))
        rng = random.Random(19)
        for _ in range(200):
            layout = sample_layout(H36M,rng)
            self.assertIn("left_elbow",layout.spec.names)
            self.assertIn("right_elbow",layout.spec.names)
            self.assertTrue(all(n in TRAIN_DROP for n in H36M.names if n not in layout.spec.names))
            allowed_midpoints = {a+"__"+b+"_mid" for a,b in TRAIN_EDGES}
            self.assertTrue(set(layout.spec.names)-set(H36M.names) <= allowed_midpoints)
            self.assertTrue(torch.isfinite(layout.apply(x)).all())

    def test_same_weights_variable_j_and_target(self):
        model = self.small_model().eval()
        x = torch.randn(2,5,17,3)
        with torch.no_grad():
            ref = model(x,skeleton=H36M)
            for name in ("canonical","holdout_sparse","holdout_dense","holdout_mixed"):
                l = eval_layout(H36M,name)
                obs = l.apply(x)
                p = model(obs,skeleton=l.spec)
                native = model(obs,skeleton=l.spec,target_skeleton=l.spec)
                self.assertEqual(p.shape,(2,5,17,3))
                self.assertEqual(native.shape,(2,5,len(l.spec.names),3))
                self.assertTrue(torch.isfinite(p).all() and torch.isfinite(native).all())
            perm = eval_layout(H36M,"permuted")
            torch.testing.assert_close(model(perm.apply(x),skeleton=perm.spec),ref,rtol=2e-5,atol=2e-6)
            # Metadata and coordinates in the new order must yield the same result.
            target = H36M.reindex(list(reversed(range(17))))
            torch.testing.assert_close(model(x,skeleton=H36M,target_skeleton=target),
                                       ref.flip(2),rtol=2e-5,atol=2e-6)
            self.assertEqual(model(x[...,:2],skeleton=H36M).shape,ref.shape)
            # Truly new reference node, not present in the training layout catalog.
            unknown = SkeletonSpec(("left_heel","pelvis","left_hip","left_ankle"),
                                   (3,-1,1,2), ((.14,-.90,-.08),(0.,0.,0.),(.13,-.05,0.),(.14,-.88,0.)))
            self.assertEqual(model(torch.randn(1,5,4,3),skeleton=unknown).shape,(1,5,17,3))

    def test_backward_and_state_dict(self):
        model = self.small_model().train()
        x = torch.randn(2,5,17,3)
        l = eval_layout(H36M,"holdout_sparse")
        p = model(l.apply(x),skeleton=l.spec)
        loss = (p-torch.randn_like(p)).square().mean()
        loss.backward()
        for name in ("layers.0.att_spatial.mixer.distance_q.weight",
                     "layers.0.graph_spatial.mixer.V.weight",
                     "readout.attention.in_proj_weight","anatomy_embed.0.weight","head.weight"):
            grad = dict(model.named_parameters())[name].grad
            self.assertIsNotNone(grad,name)
            self.assertTrue(torch.isfinite(grad).all(),name)
            self.assertGreater(grad.abs().sum().item(),0.,name)
        clone = self.small_model().eval()
        clone.load_state_dict(model.state_dict(),strict=True)
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(x),clone(x))
        self.assertEqual(model(x,return_rep=True).shape,(2,5,17,32))

    def test_real_xs_dimensions(self):
        model = SkeletonAGFormer().eval()
        x = torch.randn(1,27,17,3)
        with torch.no_grad():
            p = model(x)
            self.assertEqual(p.shape,(1,27,17,3))
            self.assertTrue(torch.isfinite(p).all())
            l = eval_layout(H36M,"holdout_dense")
            self.assertEqual(model(l.apply(x),skeleton=l.spec).shape,p.shape)


if __name__ == "__main__":
    unittest.main()
