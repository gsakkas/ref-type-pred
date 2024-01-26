module Ex10_6 where

import Data.Set hiding (elems)

{-@ LIQUID "--no-termination" @-}

{-@ type TRUE = {v:Bool | v } @-}
{-@ type FALSE = {v:Bool | not v} @-}

{-@ die :: {v:String | false} -> a @-}
die msg = error msg

{-@ data Map k v = Node { key   :: k
                        , value :: v
                        , left  :: Map {v:k | v < key} v
                        , right :: Map {v:k | key < v} v }
                 | Tip
@-}
data Map k v = Node { key   :: k
                    , value :: v
                    , left  :: Map k v
                    , right :: Map k v }
             | Tip

{-@ predicate In X Xs      = Set_mem X Xs               @-}
{-@ predicate Subset X Y   = Set_sub X Y                @-}
{-@ predicate Empty  X     = Set_emp X                  @-}
{-@ predicate Union X Y Z  = X = Set_cup Y Z            @-}
{-@ predicate Union1 X Y Z = Union X (Set_sng Y) Z      @-}
{-@ predicate HasKey K M   = In K (keys M)              @-}
{-@ predicate AddKey K M N = Union1 (keys N) K (keys M) @-}

assert _ x = x

{-@ predicate Elem X Ys  = In X (elems Ys) @-}
{-@ measure elems @-}
elems []     = empty
elems (x:xs) = (singleton x) `union` (elems xs)

{-@ fresh :: <mask> @-}
fresh :: [Int] -> Int
fresh []     = 0
fresh (x:xs) = go x [] xs
  where
    go :: Int -> [Int] -> [Int] -> Int
    go x s []     = assert (lemNotElem x s) (x + 1)
    go x s (y:ys) = go (1 + max x y) (x:s) ys

{-@ lemNotElem :: x:a -> xs:[{v:a | v < x}] -> {v:Bool | v <=> not (Elem x xs)} @-}
lemNotElem :: a -> [a] -> Bool
lemNotElem x []     = True
lemNotElem x (y:ys) = lemNotElem x ys

{-@ prop_fresh :: _ -> TRUE @-}
prop_fresh :: [Int] -> Bool
prop_fresh xs = not $ fresh xs `member` elems xs