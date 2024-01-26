module Ex8_5 where

import Prelude hiding (elem)
import Data.Set

{-@ measure elts @-}
elts :: (Ord a) => [a] -> Set a
elts [] = Data.Set.empty
elts (x:xs) = singleton x `union` elts xs

{-@ elem :: <mask> @-}
elem _ [] = False
elem x (y:ys) = x == y || elem x ys