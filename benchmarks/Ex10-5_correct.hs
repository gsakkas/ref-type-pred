module Ex10_5 where

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

{-@ measure keys @-}
keys                :: (Ord k) => Map k v -> Set k
keys Tip            = empty
keys (Node k _ l r) = ks `union` kl `union` kr
  where
    kl              = keys l
    kr              = keys r
    ks              = singleton k

{-@ lemNotMem :: key:k
                 -> m:Map {k:k | k /= key} v
                 -> {v:Bool | not (HasKey key m)}
@-}
lemNotMem :: k -> Map k v -> Bool
lemNotMem _   Tip            = True
lemNotMem key (Node _ _ l r) = lemNotMem key l && lemNotMem key r

assert _ x = x

{-@ mem :: (Ord k) => k:k -> m:Map k v
                   -> {v:_ | v <=> HasKey k m} @-}
mem k' (Node k _ l r)
  | k' == k   = True
  | k' <  k   = assert (lemNotMem k' r) (mem k' l)
  | otherwise = assert (lemNotMem k' l) (mem k' r)
mem _ Tip     = False

{-@ prop_mem :: _ -> _ -> TRUE @-}
prop_mem :: (Ord k) => k -> Map k v -> Bool
prop_mem k m = not (k `member` keys m) || (k `mem` m)