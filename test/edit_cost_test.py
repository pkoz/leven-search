import unittest

from cost import Edit, EditCost, EditOp


class EditCostTest(unittest.TestCase):

    def test_edit_cost_from_edit(self):
        edit = Edit('a', 'b')
        edit_cost = EditCost.from_edit(edit, 1)
        self.assertEqual(edit_cost.l1, edit.l1)
        self.assertEqual(edit_cost.l2, edit.l2)
        self.assertEqual(edit_cost.cost, 1)

    def test_edit_cost_repr(self):
        edit_cost = EditCost('a', 'b', 1)
        expected_repr = "a -> b : 1"
        self.assertEqual(expected_repr, edit_cost.__repr__())

    def test_edit_cost_repr_with_add(self):
        edit_cost = EditCost(EditOp.ADD, 'b', 1)
        expected_repr = "[+] b : 1"
        self.assertEqual(expected_repr, edit_cost.__repr__())

    def test_edit_cost_repr_with_delete(self):
        edit_cost = EditCost(EditOp.DELETE, 'b', 1)
        expected_repr = "[-] b : 1"
        self.assertEqual(expected_repr, edit_cost.__repr__())
