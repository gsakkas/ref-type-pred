module Ex12_1 where

{-@ LIQUID "--no-termination"      @-}

data AVL a =
    Leaf
    | Node { key :: a -- value
            , l :: AVL a -- left subtree
            , r :: AVL a -- right subtree
            , ah :: Int -- height
            }
    deriving (Show)

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

{-@ data AVL a = Leaf
               | Node { key :: a
                      , l   :: AVLL a key
                      , r   :: {v:AVLR a key | isBal l v 1}
                      , ah  :: {v:Nat        | isReal v l r}
                      }
@-}

-- | Trees of height N
{-@ type AVLN a N = {v: AVL a | realHeight v = N} @-}
-- | Trees of height equal to that of another T
{-@ type AVLT a T = AVLN a {realHeight T} @-}

{-@ empty :: AVLN a 0 @-}
empty = Leaf

{-@ singleton :: a -> AVLN a 1 @-}
singleton x = Node x empty empty 1

{-@ test1 :: AVLN a 1 @-}
test1 :: (Num a) => AVL a
test1 = singleton 1