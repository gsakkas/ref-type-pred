module Ex12_2 where

{-@ LIQUID "--no-termination" @-}

{-@ data AVL a = Leaf
               | Node { key :: a
                      , l    :: AVLL a key
                      , r    :: {v:AVLR a key | isBal l v 1}
                      , ah   :: {v:Nat        | isReal v l r}
                      }
@-}
data AVL a = Leaf
           | Node { key :: a -- value
           , l :: AVL a -- left subtree
           , r :: AVL a -- right subtree
           , ah :: Int -- height
           } deriving (Show)

-- | Trees with value less than X
{-@ type AVLL a X = AVL {v:a | v < X} @-}

-- | Trees with value greater than X
{-@ type AVLR a X = AVL {v:a | X < v} @-}

{-@ measure realHeight @-}
realHeight :: AVL a -> Int
realHeight Leaf = 0
realHeight (Node _ l r _) = nodeHeight l r

{-@ inline nodeHeight @-}
nodeHeight l r = 1 + max' hl hr
                    where
                        hl = realHeight l
                        hr = realHeight r

{-@ inline max' @-}
max' :: Int -> Int -> Int
max' x y = if x > y then x else y

{-@ inline isReal @-}
isReal v l r = v == nodeHeight l r

{-@ inline isBal @-}
isBal l r n = 0 - n <= d && d <= n
                where
                    d = realHeight l - realHeight r

-- | Trees of height N
{-@ type AVLN a N = {v: AVL a | realHeight v = N} @-}
-- | Trees of height equal to that of another T
{-@ type AVLT a T = AVLN a {realHeight T} @-}

{-@ empty :: AVLN a 0 @-}
empty = Leaf

{-@ mkNode :: <mask> @-}
mkNode :: a -> AVL a -> AVL a -> AVL a
mkNode v l r = Node v l r h
    where
        h = 1 + max' hl hr
        hl = getHeight l
        hr = getHeight r

{-@ measure getHeight @-}
{-@ getHeight :: t:_ -> {v:Nat | v = realHeight t} @-}
getHeight Leaf = 0
getHeight (Node _ _ _ n) = n

{-@ inline leftBig @-}
leftBig l r = diff l r == 2

{-@ inline rightBig @-}
rightBig l r = diff r l == 2

{-@ inline diff @-}
diff s t = getHeight s - getHeight t

{-@ measure balFac @-}
balFac Leaf           = 0
balFac (Node _ l r _) = getHeight l - getHeight r

{-@ inline leftHeavy @-}
leftHeavy t = balFac t > 0

{-@ inline rightHeavy @-}
rightHeavy t = balFac t < 0

{-@ inline noHeavy @-}
noHeavy t = balFac t == 0

{-@ balL0 :: x:a
          -> l:{AVLL a x | noHeavy l}
          -> r:{AVLR a x | leftBig l r}
          -> AVLN a { realHeight l + 1}
@-}
balL0 v (Node lv ll lr _) r = mkNode lv ll (mkNode v lr r)

{-@ balLL :: x:a
          -> l:{AVLL a x | leftHeavy l}
          -> r:{AVLR a x | leftBig l r}
          -> AVLT a l
@-}
balLL v (Node lv ll lr _) r = mkNode lv ll (mkNode v lr r)

{-@ balLR :: x:a
          -> l:{AVLL a x | rightHeavy l}
          -> r:{AVLR a x | leftBig l r}
          -> AVLT a l
@-}
balLR v (Node lv ll (Node lrv lrl lrr _) _) r = mkNode lrv (mkNode lv ll lrl) (mkNode v lrr r)

{-@ balR0 :: x:a
          -> l:AVLL a x
          -> r:{AVLR a x | rightBig l r && noHeavy r}
          -> AVLN a { realHeight r + 1}
@-}
balR0 v l (Node rv rl rr _) = mkNode rv (mkNode v l rl) rr

{-@ balRR :: x:a
          -> l:AVLL a x
          -> r:{AVLR a x | rightBig l r && rightHeavy r}
          -> AVLT a r
@-}
balRR v l (Node rv rl rr _) = mkNode rv (mkNode v l rl) rr

{-@ balRL :: x:a
          -> l:AVLL a x
          -> r:{AVLR a x | rightBig l r && leftHeavy r}
          -> AVLT a r
@-}
balRL v l (Node rv (Node rlv rll rlr _) rr _) = mkNode rlv (mkNode v l rll) (mkNode rv rlr rr)

{-@ inline reBal @-}
reBal l r t = bigHt l r t && balHt l r t

{-@ inline balHt @-}
balHt l r t = not (isBal l r 1) || isReal (realHeight t) l r

{-@ inline eqOrUp @-}
eqOrUp s t = d == 0 || d == 1
    where
        d = diff t s

{-@ inline bigHt @-}
bigHt l r t = lBig && rBig
    where
    lBig = hl < hr || eqOrUp l t
    rBig = hl >= hr || eqOrUp r t
    hl   = realHeight l
    hr   = realHeight r

{-@ bal :: x:a
        -> l:AVLL a x
        -> r:{AVLR a x | isBal l r 2}
        -> {t:AVL a | reBal l r t}
@-}
bal :: a -> AVL a -> AVL a -> AVL a
bal v l r
    | isLeftBig && leftHeavy l   = balLL v l r
    | isLeftBig && rightHeavy l  = balLR v l r
    | isLeftBig                  = balL0 v l r
    | isRightBig && leftHeavy r  = balRL v l r
    | isRightBig && rightHeavy r = balRR v l r
    | isRightBig                 = balR0 v l r
    | otherwise                  = mkNode v l r
    where
        isLeftBig  = leftBig l r
        isRightBig = rightBig l r