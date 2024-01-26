module Ex8_5 where

import Prelude hiding (elem)
import Data.Set

{-@ measure elts @-}
elts :: (Ord a) => [a] -> Set a
elts [] = Data.Set.empty
elts (x:xs) = singleton x `union` elts xs

{-@ type TRUE = {v:Bool | v} @-}
{-@ type FALSE = {v:Bool | not v} @-}

{-@ elem :: <mask> @-}
elem :: (Eq a) => a -> [a] -> Bool
elem _ [] = False
elem x (y:ys) = x == y || elem x ys

{-@ test1 :: TRUE @-}
test1 = elem 2 [1, 2, 3]
test1' = test1 == True

{-@ test2 :: FALSE @-}
test2 = elem 2 [1, 3]
test2' = test2 == False