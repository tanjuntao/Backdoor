# Code from
# https://github.com/AharonSambol/PrettyPrintTree

import os
import re
import sys
from collections.abc import Iterable
from typing import Callable, List, Dict, Optional
from colorama import Back, Style

from linkefl.vfl.tree.core.decision_tree import DecisionTree
from linkefl.vfl.tree.core.tree_node import _DecisionNode


if sys.version_info < (3, 7):
    sys.exit("Sorry, Python < 3.7 is not supported")


class PrintTree:
    @classmethod
    def plot_tree_strs(
        cls,
        tree_strs: Optional[Dict[str, str]] = None,
        trees: Optional[List[DecisionTree]] = None,
        tree_structure: str = "VERTICAL",
        file_dir: str = "./models",
    ):
        file_path = os.path.join(file_dir, "trees.txt")

        if tree_strs is None:
            tree_strs = cls.get_tree_strs(trees, tree_structure)

        with open(file_path, "a") as f:
            for tree_id, tree_str in enumerate(tree_strs.values(), 1):
                f.write(f"Tree{tree_id}:\n")
                f.write(tree_str)
                f.write("\n\n")

    @classmethod
    def get_tree_strs(
        cls,
        trees: List[DecisionTree],
        tree_structure: str = "VERTICAL",
    ) -> Dict[str, str]:
        assert tree_structure in [
            "VERTICAL",
            "HORIZONTAL",
        ], "tree_structure should be VERTICAL or HORIZONTAL"

        if tree_structure == "HORIZONTAL":
            orientation = PrettyPrintTree.HORIZONTAL
        else:
            orientation = PrettyPrintTree.VERTICAL

        pt = PrettyPrintTree(
            get_children=lambda x: x.children if x else [],
            get_val=lambda x: x.print_val if x else "",
            default_orientation=orientation,
            border=True,
            color=None,
            return_instead_of_print=True,
        )

        tree_strs = {}
        for tree_id, tree in enumerate(trees, 1):
            cls._prepare_print_val(tree, tree.root)
            tree_str = pt(tree.root)
            tree_strs[f"tree{tree_id}"] = tree_str
        return tree_strs

    @classmethod
    def _prepare_print_val(
        cls,
        tree: DecisionTree,
        node: _DecisionNode,
    ) -> None:
        if not node:
            return

        if node.value is not None:
            # leaf node
            if isinstance(node.value, (int, float)):
                print_val = f"value: {node.value: .3f}"
            else:
                # the leaf value of multi class tree is a list.
                print_val = f"value: {node.value[0]: .3f}, .."
        else:
            # mid node
            if node.party_id == 0:
                print_val = "active_party\n"
                print_val += f"record_id: {node.record_id}\n"
                print_val += f"feature: f{int(tree.record[node.record_id][0])}\n"
                print_val += f"threshold: {tree.record[node.record_id][1]: .3f}"
            else:
                print_val = f"passive_party_{node.party_id}\n"
                print_val += f"record_id: {node.record_id}\n"
                print_val += "feature: encrypt\n"
                print_val += "threshold: encrypt"

        node.print_val = print_val
        node.children = []

        if node.left_branch:
            cls._prepare_print_val(tree, node.left_branch)
            node.children.append(node.left_branch)
        if node.right_branch:
            cls._prepare_print_val(tree, node.right_branch)
            node.children.append(node.right_branch)


class DictTree:
    def __init__(self, val, children=None):
        if children is None:
            if isinstance(val, Iterable) and not isinstance(val, str):
                children = val
                val = str(type(val)).removeprefix("<class '").removesuffix("'>").upper()
            else:
                children = []

        self.val = val
        if isinstance(children, dict):
            self.children = [DictTree(v, c) for v, c in children.items()]
        elif isinstance(children, Iterable) and not isinstance(children, str):
            self.children = [DictTree(x) for x in children]
        else:
            self.children = [children]


class PrettyPrintTree:
    VERTICAL = False
    HORIZONTAL = True

    def __init__(
        self,
        get_children: Callable[[object], Iterable] = None,
        get_val: Callable[[object], object] = None,
        get_label: Callable[[object], object] = None,
        *,
        label_color="",
        show_newline_literal: bool = False,
        return_instead_of_print: bool = False,
        trim=False,
        start_message: Callable[[object], str] = None,
        color: Optional[str]=Back.LIGHTBLACK_EX,
        border: bool = False,
        max_depth: int = -1,
        default_orientation: bool = False,
    ):
        # this is a lambda which returns a list of all the children
        # in order to support trees of different kinds eg:
        #   self.child_right, self.child_left... or
        #   self.children = []... or
        #   self.children = {}... or anything else

        self.get_children = (
            (lambda x: x.children) if get_children is None else get_children
        )
        self.get_node_val = (lambda x: x.value) if get_val is None else get_val
        self.get_label = get_label
        self.label_color = label_color
        # only display first x chars
        self.trim = trim
        # if true will display \n as \n and not as new lines
        self.show_newline = show_newline_literal
        self.dont_print = return_instead_of_print
        self.start_message = start_message
        self.color = color
        self.border = border
        self.max_depth = max_depth
        self.default_orientation = default_orientation

    def print_json(self, dic, name="JSON", max_depth: int = 0):
        if max_depth:
            self.max_depth = max_depth
        self.get_children = lambda x: x.children if isinstance(x, DictTree) else []
        self.get_node_val = lambda x: x.val if isinstance(x, DictTree) else x
        self(DictTree(name, dic))

    def __call__(self, node, max_depth: int = 0, orientation=None):
        if orientation is not None:
            self.default_orientation = orientation
        if (
            self.default_orientation == PrettyPrintTree.HORIZONTAL
            and self.get_label is not None
        ):
            print(
                """  ^                                                                    ^
 /^\\                                                                  /^\\
 /^\\                                                                  /^\\
//|\\\\     As of now labels are only supported for vertical trees     //|\\\\
//|\\\\                                                                //|\\\\
  |                                                                    |
""",
            )
        if isinstance(node, dict) or isinstance(node, list) or isinstance(node, tuple):
            self.print_json(node, max_depth=max_depth)
            return
        if self.start_message is not None and not self.dont_print:
            print(self.start_message(node))
        if max_depth:
            self.max_depth = max_depth
        if self.default_orientation == PrettyPrintTree.HORIZONTAL:
            res = self.tree_to_str_horizontal(node)
        else:
            res = self.tree_to_str(node)

        is_node = lambda x: (
            x.startswith("[")
            or x.startswith("|")
            or (x.startswith("│") and x.strip() != "│")
            or len(x) > 1
            and x[1:-1] == "─" * (len(x) - 2)
            and x[0] + x[-1] in ["┌┐", "└┘"]
        )
        lines = [
            "".join(self.color_txt(x) if is_node(x) else x for x in line)
            for line in res
        ]
        if self.dont_print:
            if self.start_message:
                return self.start_message(node) + "\n" + "\n".join(lines)
            return "\n".join(lines)
        print("\n".join(lines))

    def get_val(self, node):
        st_val = str(self.get_node_val(node))
        if self.trim and len(st_val) > self.trim:
            st_val = st_val[: self.trim] + "..."
        if self.show_newline:
            escape_newline = lambda match: "\\n" if match.group(0) == "\n" else "\\\\n"
            st_val = re.sub(r"(\n|\\n)", escape_newline, st_val)
        if "\n" not in st_val:
            return [[st_val]]
        lst_val = st_val.split("\n")
        longest = max(len(x) for x in lst_val)
        return [[f'{x}{" " * (longest - len(x))}'] for x in lst_val]

    def tree_to_str(self, node, depth=0):
        val = self.get_val(node)
        children = self.get_children(node)
        if not isinstance(children, list):
            children = list(children)
        if len(children) == 0:
            if len(val) == 1:
                res = [[f"[{ val[0][0] }]"]]
                if self.get_label:
                    self.put_label(node, res)
                return res
            res = self.format_box("", val)
            if self.get_label:
                self.put_label(node, res)
            return res
        to_print = [[]]
        spacing = 0
        if depth + 1 != self.max_depth:
            for child in children:
                child_print = self.tree_to_str(child, depth=depth + 1)
                for l, line in enumerate(child_print):
                    if l + 1 >= len(to_print):
                        to_print.append([])
                    if l == 0:
                        len_line = len("".join(line))
                        middle_of_child = len_line - sum(divmod(len(line[-1]), 2))
                        len_to_print_0 = len("".join(to_print[0]))
                        to_print[0].append(
                            (spacing - len_to_print_0 + middle_of_child) * " " + "┬",
                        )
                    to_print[l + 1].append(
                        " " * (spacing - len("".join(to_print[l + 1]))),
                    )
                    to_print[l + 1].extend(line)
                spacing = max(len("".join(x)) for x in to_print) + 1

            if len(to_print[0]) != 1:
                new_lines = "".join(to_print[0])
                space_before = len(new_lines) - len(new_lines.strip())
                new_lines = new_lines.strip()
                ln_of_stripped = len(new_lines)
                new_lines = (
                    " " * space_before + "┌" + new_lines[1:-1].replace(" ", "─") + "┐"
                )
                pipe_pos = middle = len(new_lines) - sum(divmod(ln_of_stripped, 2))
                new_ch = {"─": "┴", "┬": "┼", "┌": "├", "┐": "┤"}[new_lines[middle]]
                new_lines = new_lines[:middle] + new_ch + new_lines[middle + 1 :]
                to_print[0] = [new_lines]
            else:
                to_print[0][0] = to_print[0][0][:-1] + "│"
                pipe_pos = len(to_print[0][0]) - 1
            spacing = " " * (pipe_pos - sum(divmod(len(val[0][0]), 2)))
        else:
            spacing = ""

        if len(val) == 1:
            val = [[spacing, f"[{val[0][0]}]"]]
        else:
            val = self.format_box(spacing, val)
        to_print = val + to_print

        if self.get_label:
            self.put_label(node, to_print)
        return to_print

    def put_label(self, node, res):
        label = self.get_label(node)
        label = ("|" + str(label) + "|") if label else "│"
        if len(label) > len(res[0][-1]):
            diff = len(label) - len(res[0][-1])
            for row in res:
                row[0] = " " * diff + row[0]
        if len(res[0]) == 2:
            d, m = divmod(len(res[0][-1]), 2)
            pos = len(res[0][0]) + d - (1 * (1 - m))
        else:
            pos = len(res[0][0]) // 2
        res.insert(0, [" " * pos, "│"])
        res.insert(0, [" " * (pos - len(label) // 2), label])

    def tree_to_str_horizontal(self, node, depth=0):
        val = self.get_val(node)
        children = self.get_children(node)
        if not isinstance(children, list):
            children = list(children)
        if len(children) == 0:
            if len(val) == 1:
                return [["-", "[" + val[0][0] + "]"]]
            box = self.format_box("", val)
            box[0][0] = "-"
            for i in range(1, len(box)):
                box[i][0] = " "
            return box
        to_print = []
        if depth + 1 != self.max_depth:
            for i, child in enumerate(children):
                if i != 0:
                    to_print.extend([[]])
                child_print = self.tree_to_str_horizontal(child, depth=depth + 1)
                to_print.extend(child_print)

        val_width = max(len("".join(x)) for x in val) + 2
        middle_children = sum(divmod(len(to_print), 2))
        middle_this = sum(divmod(len(val), 2))
        pos = max(0, middle_children - middle_this)
        first = last = -1
        added = False
        for r, row in enumerate(to_print[:pos]):
            if len("".join(row)) > 0 and "".join(row)[0] == "-":
                while row[0] == "":
                    row.pop(0)
                row[0] = ("┌" if first == -1 else "├") + row[0][1:]
                if first == -1:
                    first = r
                last = r
            elif first != -1:
                if row:
                    row[0] = "│" + row[0][1:]
                else:
                    row.append("│")
            row.insert(0, " " * (val_width + 1))  # + "├" "┬" "¬" "┼" "┤" "│" '┌'
        for r, row in enumerate(val):
            if len(to_print) == r + pos:
                to_print.append([])
            if "".join(to_print[r + pos]).startswith("-"):
                while to_print[r + pos][0] == "":
                    to_print[r + pos].pop(0)
                str_added = str(added)
                str_firstbool = str(first == -1)
                symbol = {"TT": "┌", "TF": "├", "FT": "┬", "FF": "┼"}[
                    str_added[0] + str_firstbool[0]
                ]
                to_print[r + pos] = [symbol] + to_print[r + pos][1:]
                if first == -1:
                    first = r + pos
                last = r + pos
            else:
                if to_print[r + pos]:
                    to_print[r + pos][0] = ("┤" if not added else "│") + to_print[
                        r + pos
                    ][0][1:]
                elif depth + 1 != self.max_depth:
                    to_print[r + pos].append("┤" if not added else "│")
                else:
                    to_print[r + pos].append("")
                if not added:
                    last = r + pos
            added = True
            to_print[r + pos].insert(0, "[" + "".join(row) + "]")
            if r + 1 == sum(divmod(len(val), 2)):
                to_print[r + pos].insert(0, "-")
            else:
                to_print[r + pos].insert(0, " ")
            to_print[r + pos].insert(
                2,
                "  " * (val_width - len("[" + "".join(row) + "]")),
            )
        for r, row in enumerate(to_print[pos + len(val) :]):
            if len("".join(row)) > 0 and "".join(row)[0] == "-":
                while row[0] == "":
                    row.pop(0)
                row[0] = "├" + row[0][1:]
                last = r + pos + len(val)
            else:
                if row:
                    row[0] = "│" + row[0][1:]
                else:
                    row.append("│")
            row.insert(0, " " * (val_width + 1))
        indx = 0
        while to_print[last][indx].strip() != "" or to_print[last][indx + 1].startswith(
            "[",
        ):
            indx += 1
        indx += 1
        to_print[last][indx] = {
            "┬": "─",
            "├": "└",
            "┌": "─",
            "┼": "┴",
            "┤": "┘",
            "": "",
        }[to_print[last][indx]]
        for i in range(last + 1, len(to_print)):
            indx = 0
            while to_print[i][indx].strip() != "" or to_print[i][indx + 1].startswith(
                "[",
            ):
                indx += 1
            indx += 1
            to_print[i][indx] = " " + to_print[i][indx][1:]
        return to_print

    def color_txt(self, x):
        spaces = " " * (len(x) - len(x.lstrip()))
        x = x.lstrip()
        is_label = x.startswith("|")
        if is_label:
            x = f" { x[1:-1] } "
            x = (self.label_color + x + Style.RESET_ALL) if self.label_color else x
            return spaces + x
        txt = x if self.border else f" { x[1:-1] } "
        txt = (self.color + txt + Style.RESET_ALL) if self.color else txt
        return spaces + txt

    def format_box(self, spacing, val):
        for r, row in enumerate(val):
            val[r] = [spacing, f"│{row[0]}│"]
        if self.border:
            top = [[spacing, "┌" + "─" * (len(val[0][1]) - 2) + "┐"]]
            bottom = [[spacing, "└" + "─" * (len(val[0][1]) - 2) + "┘"]]
            return top + val + bottom
        return
