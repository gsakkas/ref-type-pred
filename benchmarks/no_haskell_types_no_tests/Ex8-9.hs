module Ex8_9 where

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

{-@ reverse :: xs:UList a -> UList a @-}
reverse = rev' []
  where
    {-@ rev' :: <mask> a @-}
    rev' a []     = a
    rev' a (x:xs) = rev' (x:a) xs