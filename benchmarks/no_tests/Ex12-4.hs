module Ex12_4 where

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

{-@ mkNode :: x:a -> l:AVLL a x -> {r:AVLR a x | isBal l r 1} -> AVLN a {nodeHeight l r} @-}
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

{-@ balRR :: <mask> @-}
balRR :: a -> AVL a -> AVL a -> AVL a
balRR v l (Node rv rl rr _) = mkNode rv (mkNode v l rl) rr