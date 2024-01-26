module Ex8_4 where

import Data.Set hiding (size)

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

{-@ measure size @-}
{-@ size :: [a] -> Nat @-}
size :: [a] -> Int
size [] = 0
size (_:rs) = 1 + size rs

{-@ measure fst @-}
fst (x, _) = x
{-@ measure snd @-}
snd (_, y) = y

{-@ predicate Sum2 X N = size (fst X) + size (snd X) = N @-}

{-@ append' :: xs:_ -> ys:_ -> ListUn a xs ys @-}
append' [] ys = ys
append' (x:xs) ys = x : append' xs ys

{-@ halve :: <mask> @-}
halve :: Int -> [a] -> ([a], [a])
halve 0 xs = ([], xs)
halve n (x:y:zs) = (x:xs, y:ys) where (xs, ys) = halve (n-1) zs
halve _ xs = ([], xs)