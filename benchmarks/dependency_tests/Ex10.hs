module Ex10 where

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

{-@ emp :: <mask_1> @-}
emp :: Map k v
emp     = Tip

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

{-@ plus :: Val -> Val -> Val @-}
plus (Const i) (Const j) = Const (i+j)
plus _         _         = die "Bad call to plus"

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

{-@ lemNotMem :: <mask_2> @-}
lemNotMem :: k -> Map k v -> Bool
lemNotMem _   Tip            = True
lemNotMem key (Node _ _ l r) = lemNotMem key l && lemNotMem key r

assert _ x = x

{-@ prop_mem :: _ -> _ -> TRUE @-}
prop_mem :: (Ord k) => k -> Map k v -> Bool
prop_mem k m = not (k `member` keys m) || (k `mem` m)

{-@ mem :: <mask_3> @-}
mem :: Ord k => k -> Map k v -> Bool
mem k' (Node k _ l r)
  | k' == k   = True
  | k' <  k   = assert (lemNotMem k' r) (mem k' l)
  | otherwise = assert (lemNotMem k' l) (mem k' r)
mem _ Tip     = False

{-@ get :: <mask_4> @-}
get :: (Ord k) => k -> Map k v -> v
get k' (Node k v l r)
  | k' == k   = v
  | k' <  k   = assert (lemNotMem k' r) $
                  get k' l
  | otherwise = assert (lemNotMem k' l) $
                  get k' r
get _ Tip     = die "Lookup failed? Impossible."

{-@ set :: <mask_5> @-}
set :: (Ord k) => k -> v -> Map k v -> Map k v
set k' v' (Node k v l r)
  | k' == k   = Node k v' l r
  | k' <  k   = Node k v (set k' v l) r
  | otherwise = Node k v l (set k' v r)
set k' v' Tip = Node k' v' Tip Tip

{-@ type Env = Map Var Val @-}
{-@ type ClosedExpr G = {v:Expr | Subset (free v) (keys G)} @-}

{-@ eval :: <mask_6> @-}
eval :: Map Var Expr -> Expr -> Expr
eval _ i@(Const _)   = i
eval g (Var x)       = get x g
eval g (Plus e1 e2)  = plus  (eval g e1) (eval g e2)
eval g (Let x e1 e2) = eval g' e2
  where
    g'               = set x v1 g
    v1               = eval g e1

{-@ topEval :: <mask_7> @-}
topEval     = eval emp

{-@ evalAny :: <mask_8> @-}
evalAny :: Map Var Expr -> Expr -> Maybe Expr
evalAny g e
  | ok        = Just $ eval g e
  | otherwise = Nothing
  where
    ok        = isSubsetOf (free e) (keys g)

{-@ predicate Elem X Ys  = In X (elems Ys) @-}
{-@ measure elems @-}
elems []     = empty
elems (x:xs) = (singleton x) `union` (elems xs)

{-@ lemNotElem :: <mask_9> @-}
lemNotElem :: a -> [a] -> Bool
lemNotElem x []     = True
lemNotElem x (y:ys) = lemNotElem x ys

{-@ prop_fresh :: _ -> TRUE @-}
prop_fresh :: [Int] -> Bool
prop_fresh xs = not $ fresh xs `member` elems xs

{-@ fresh :: <mask_10> @-}
fresh :: [Int] -> Int
fresh []     = 0
fresh (x:xs) = go x [] xs
  where
    go :: Int -> [Int] -> [Int] -> Int
    go x s []     = assert (lemNotElem x s) (x + 1)
    go x s (y:ys) = go (1 + max x y) (x:s) ys