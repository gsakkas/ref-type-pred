module Ex8_10 where

import Prelude hiding (reverse)
import Data.Set hiding (filter)

{-@ type Nat = {v:Int | 0 <= v} @-}

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

{-@ measure unique @-}
unique :: (Ord a) => [a] -> Bool
unique [] = True
unique (x:xs) = unique xs && not (member x (elts xs))

{-@ type UList a = {v:[a] | unique v }@-}
{-@ predicate In X Xs = Set_mem X (elts Xs) @-}

{-@ isin :: x:_ -> ys:_ -> {v:Bool | v <=> In x ys} @-}
isin x (y:ys)
  | x == y    = True
  | otherwise = x `isin` ys
isin _ [] = False

{-@ nub :: (Eq a) => [a] -> UList a @-}
nub :: (Eq a) => [a] -> [a]
nub xs = nubAcc [] xs
  where
    nubAcc seen [] = seen
    nubAcc seen (x:xs)
      | x `isin` seen = nubAcc seen xs
      | otherwise     = nubAcc (x:seen) xs

test1 :: [Int]
{-@ test1 :: UList Int @-}
test1 = nub [1, 2, 3]
test1' = test1 == [1, 2, 3]

test2 :: [Int]
{-@ test2 :: UList Int @-}
test2 = nub [1, 2, 3]
test2' = test2 == [3, 2, 1]

test3 :: [Int]
{-@ test3 :: UList Int @-}
test3 = nub [1, 2, 2, 3, 3]
test3' = test3 == [3, 2, 1]