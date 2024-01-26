module Ex8_3 where

import Data.Set

{-@ measure elts @-}
elts :: (Ord a) => [a] -> Set a
elts [] = Data.Set.empty
elts (x:xs) = singleton x `union` elts xs

{-@ type ListS a S = {v:[a] | elts v = S} @-}
{-@ type ListEmp a = ListS a {Set_empty 0} @-}
{-@ type ListEq a X = ListS a {elts X} @-}
{-@ type ListSub a X = {v:[a]| Set_sub (elts v) (elts X)} @-}
{-@ type ListUn a X Y = ListS a {Set_cup (elts X) (elts Y)} @-}
{-@ type ListUn1 a X Y = ListS a {Set_cup (Set_sng X) (elts Y)} @-}

{-@ reverse' :: xs:[a] -> ListEq a xs @-}
reverse' xs = revHelper [] xs

{-@ revHelper :: x:[a] -> y:[a] -> r:ListUn a x y @-}
revHelper :: [a] -> [a] -> [a]
revHelper acc [] = acc
revHelper acc (x:xs) = revHelper (x:acc) xs

test1 = reverse' [1, 2, 3] == [3, 2, 1]
test2 = reverse' ["apple", "lemon", "orange"] == ["orange", "lemon", "apple"]