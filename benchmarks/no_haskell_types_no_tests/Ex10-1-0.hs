module Ex10_1_0 where

import Data.Set hiding (elems)

{-@ LIQUID "--no-termination" @-}

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

{-@ emp :: {m:Map k v | Empty (keys m)} @-}
emp     = Tip

{-@ set :: (Ord k) => k:k -> v -> m:Map k v
                   -> {n: Map k v | AddKey k m n} @-}
set k' v' (Node k v l r)
  | k' == k   = Node k v' l r
  | k' <  k   = Node k v (set k' v l) r
  | otherwise = Node k v l (set k' v r)
set k' v' Tip = Node k' v' Tip Tip

{-@ measure val @-}
val              :: Expr -> Bool
val (Const _)    = True
val (Var _)      = False
val (Plus _ _)   = False
val (Let _ _ _ ) = False

type Var  = String
data Expr = Const Int
          | Var   Var
          | Plus  Expr Expr
          | Let   Var  Expr Expr

{-@ type Val = {v:Expr | val v} @-}

{-@ plus                 :: Val -> Val -> Val @-}
plus (Const i) (Const j) = Const (i+j)
plus _         _         = die "Bad call to plus"

{-@ type Env = Map Var Val @-}
{-@ type ClosedExpr G = {v:Expr | Subset (free v) (keys G)} @-}

{-@ eval :: <mask> @-}
eval _ i@(Const _)   = i
eval g (Var x)       = get x g
eval g (Plus e1 e2)  = plus  (eval g e1) (eval g e2)
eval g (Let x e1 e2) = eval g' e2
  where
    g'               = set x v1 g
    v1               = eval g e1

{-@ topEval :: {v:Expr | Empty (free v)} -> Val @-}
topEval     = eval emp

{-@ measure free @-}
free               :: Expr -> Set Var
free (Const _)     = empty
free (Var x)       = singleton x
free (Plus e1 e2)  = xs1 `union`  xs2
  where
    xs1            = free e1
    xs2            = free e2
free (Let x e1 e2) = xs1 `union` (xs2 `difference` xs)
  where
    xs1            = free e1
    xs2            = free e2
    xs             = singleton x

{-@ lemNotMem :: key:k
                 -> m:Map {k:k | k /= key} v
                 -> {v:Bool | not (HasKey key m)}
@-}
lemNotMem :: k -> Map k v -> Bool
lemNotMem _   Tip            = True
lemNotMem key (Node _ _ l r) = lemNotMem key l && lemNotMem key r

{-@ get :: (Ord k) => k:k -> m:{Map k v | HasKey k m} -> v @-}
get k' (Node k v l r)
  | k' == k   = v
  | k' <  k   = assert (lemNotMem k' r) $
                  get k' l
  | otherwise = assert (lemNotMem k' l) $
                  get k' r
get _ Tip     = die "Lookup failed? Impossible."

assert _ x = x