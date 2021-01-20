#include "OneLvPixMix.h"

#include <numeric>

#include "../diminished_reality/cv_utils.h"

OneLvPixMix::OneLvPixMix()
    : borderSize(2), borderSizePosMap(1), windowSize(5), toLeft(0, -1), toRight(0, 1), toUp(-1, 0), toDown(1, 0) // borderSize = windowSize / 2
{
    vSptAdj = {
        cv::Vec2i(-1, -1), cv::Vec2i(-1, 0), cv::Vec2i(-1, 1),
        cv::Vec2i(0, -1),                   cv::Vec2i(0, 1),
        cv::Vec2i(1, -1), cv::Vec2i(1, 0), cv::Vec2i(1, 1)
    };
}

OneLvPixMix::~OneLvPixMix() {}

void OneLvPixMix::init(
    const cv::Mat_<cv::Vec3b> &color,
    const cv::Mat_<uchar> &mask,
    const cv::Mat_<uchar> &discarding_area,
    const cv::Mat_<uchar> &gradient
)
{
    std::random_device rnd;
    mt = std::mt19937(rnd());
    cRand = std::uniform_int_distribution<int>(0, color.cols - 1);
    rRand = std::uniform_int_distribution<int>(0, color.rows - 1);

    cv::copyMakeBorder(color, mColor[W_BORDER], borderSize, borderSize, borderSize, borderSize, cv::BORDER_REFLECT);
    cv::copyMakeBorder(mask, mMask[W_BORDER], borderSize, borderSize, borderSize, borderSize, cv::BORDER_REFLECT);
    cv::copyMakeBorder(discarding_area, mDiscardings[W_BORDER], borderSize, borderSize, borderSize, borderSize, cv::BORDER_REFLECT);
    cv::copyMakeBorder(gradient, mGradient[W_BORDER], borderSize, borderSize, borderSize, borderSize, cv::BORDER_REFLECT);
    mColor[WO_BORDER] = cv::Mat(mColor[W_BORDER], cv::Rect(borderSize, borderSize, color.cols, color.rows));
    mMask[WO_BORDER] = cv::Mat(mMask[W_BORDER], cv::Rect(borderSize, borderSize, mask.cols, mask.rows));
    mDiscardings[WO_BORDER] = cv::Mat(mDiscardings[W_BORDER], cv::Rect(borderSize, borderSize, discarding_area.cols, discarding_area.rows));
    mGradient[WO_BORDER] = cv::Mat(mGradient[W_BORDER], cv::Rect(borderSize, borderSize, gradient.cols, gradient.rows));

    mPosMap[WO_BORDER] = cv::Mat_<cv::Vec2i>(mColor[WO_BORDER].size());
    for (int r = 0; r < mPosMap[WO_BORDER].rows; ++r) {
        for (int c = 0; c < mPosMap[WO_BORDER].cols; ++c) {
            if (mMask[WO_BORDER](r, c) == 0) mPosMap[WO_BORDER](r, c) = getValidRandPos();
            else mPosMap[WO_BORDER](r, c) = cv::Vec2i(r, c);
        }
    }
    cv::copyMakeBorder(mPosMap[WO_BORDER], mPosMap[W_BORDER], borderSizePosMap, borderSizePosMap, borderSizePosMap, borderSizePosMap, cv::BORDER_REFLECT);
    mPosMap[WO_BORDER] = cv::Mat(mPosMap[W_BORDER], cv::Rect(1, 1, color.cols, color.rows));

    //for(int count = 0; count < 10; count++) {
    //    std::list<std::pair<cv::Vec2i, int>> priorPos;
    //    for (int r = 0; r < mGradient[WO_BORDER].rows; ++r) {
    //        for (int c = 0; c < mGradient[WO_BORDER].cols; ++c) {
    //            if (mMask[WO_BORDER](r, c) != 0 || mDiscardings[WO_BORDER](r, c)) continue;  // É}ÉXÉNóÃàÊÇÃÇ›èàóùÇ∑ÇÈ
    //            if (mGradient[WO_BORDER](r, c) != 0) continue;

    //            int g = 0;
    //            for (int i = 0; i < windowSize; i++) {
    //                uchar* ptrGradient = mGradient[W_BORDER].ptr<uchar>(r + i);
    //                for (int j = 0; j < windowSize; j++) {
    //                    g += ptrGradient[c + j];
    //                }
    //            }

    //            if (g == 0) continue;

    //            auto smaller_than_g_itr = std::find_if(priorPos.begin(), priorPos.end(), [&](const std::pair<cv::Vec2i, int>& pos) {
    //                return pos.second < g;
    //            }); // gÇÊÇËè¨Ç≥Ç¢óvëfÇÃÉCÉeÉåÅ[É^ÇéÊìæ

    //            bool isInWindow = std::any_of(priorPos.begin(), smaller_than_g_itr, [&](const auto& pos) {
    //                if (pos.first[0] == r && pos.first[1] == c) return true;
    //                else return abs(pos.first[0] - r) < borderSize || abs(pos.first[1] - c) < borderSize;
    //            });
    //            if (isInWindow) continue;

    //            priorPos.insert(smaller_than_g_itr, std::make_pair(cv::Vec2i(r, c), g));    // ç~èáÇ…ï¿Ç◊ÇÈ
    //        }
    //    }

    //    if (priorPos.size() == 0) break;

    //    for (auto it = priorPos.begin(); it != priorPos.end(); it++) {
    //        cv::Vec2i pos = (*it).first;
    //        float minAc = std::numeric_limits<float>::max();
    //        cv::Vec2i bestPos = cv::Vec2i::all(-1);
    //        for (int r = 0; r < mGradient[WO_BORDER].rows; ++r) {
    //            for (int c = 0; c < mGradient[WO_BORDER].cols; ++c) {
    //                if (r == pos[0] && c == pos[1]) continue;

    //                float ac = calcAppCost(pos, cv::Vec2i(r, c));

    //                if (ac < minAc) {
    //                    minAc = ac;
    //                    bestPos = cv::Vec2i(r, c);
    //                }
    //            }
    //        }

    //        if (bestPos == cv::Vec2i::all(-1)) continue;

    //        for (int i = 0; i < windowSize; i++) {
    //            cv::Vec2i* ptrPosMap = mPosMap[W_BORDER].ptr<cv::Vec2i>(pos[0] + i);
    //            uchar* ptrGradient = mGradient[W_BORDER].ptr<uchar>(pos[0] + i);
    //            for (int j = 0; j < windowSize; j++) {
    //                ptrPosMap[pos[1] + j] = bestPos + cv::Vec2i(i, j);
    //                ptrGradient[pos[1] + j] = mGradient[W_BORDER](bestPos[0] + i, bestPos[1] + j);
    //            }
    //        }
    //    }
    //}

    //cv::imshow("after sobel", mGradient[WO_BORDER]);
    //cv::waitKey();
}

void OneLvPixMix::execute(
    const float scAlpha,
    const float acAlpha,
    const int maxItr,
    const int maxRandSearchItr,
    const float threshDist
)
{
    const float ccAlpha = 1.0f - scAlpha - acAlpha;
    const float thDist = std::pow(std::max(mColor[WO_BORDER].cols, mColor[WO_BORDER].rows) * threshDist, 2.0f);

    for (int itr = 0; itr < maxItr; ++itr) {
        cv::Mat_<cv::Vec3b> vizPosMap, vizColor;
        Util::createVizPosMap(mPosMap[WO_BORDER], vizPosMap);
        cv::resize(vizPosMap, vizPosMap, cv::Size(640, 480), 0.0, 0.0, cv::INTER_NEAREST);
        cv::imshow("prev posMap", vizPosMap);
        cv::resize(mColor[WO_BORDER], vizColor, cv::Size(640, 480), 0.0, 0.0, cv::INTER_NEAREST);
        cv::imshow("prev color", vizColor);
        cv::waitKey(1);

        if (itr % 2 == 0) bwdUpdate(scAlpha, acAlpha, ccAlpha, thDist, maxRandSearchItr);
        else fwdUpdate(scAlpha, acAlpha, ccAlpha, thDist, maxRandSearchItr);

        Util::createVizPosMap(mPosMap[WO_BORDER], vizPosMap);
        cv::resize(vizPosMap, vizPosMap, cv::Size(640, 480), 0.0, 0.0, cv::INTER_NEAREST);
        cv::imshow("curr posMap", vizPosMap);
        cv::resize(mColor[WO_BORDER], vizColor, cv::Size(640, 480), 0.0, 0.0, cv::INTER_NEAREST);
        cv::imshow("curr color", vizColor);
        cv::waitKey(1);

        inpaint();
    }
}

void OneLvPixMix::inpaint()
{
    for (int r = 0; r < mColor[WO_BORDER].rows; ++r) {
        cv::Vec3b *ptrColor = mColor[WO_BORDER].ptr<cv::Vec3b>(r);
        cv::Vec2i *ptrPosMap = mPosMap[WO_BORDER].ptr<cv::Vec2i>(r);
        for (int c = 0; c < mColor[WO_BORDER].cols; ++c) {
            ptrColor[c] = mColor[WO_BORDER](ptrPosMap[c]);
        }
    }
}

float OneLvPixMix::calcSptCost(
    const cv::Vec2i &target,
    const cv::Vec2i &ref,
    float maxDist,
    float w
)
{
    const float normFactor = maxDist * 2.0f;

    float sc = 0.0f;
    for (const auto &v : vSptAdj) {
        cv::Vec2f diff((ref + v) - mPosMap[W_BORDER](target + cv::Vec2i(borderSizePosMap, borderSizePosMap) + v));
        sc += std::min(diff.dot(diff), maxDist);
    }

    return sc * w / normFactor;
}

float OneLvPixMix::calcAppCost(
    const cv::Vec2i &target,
    const cv::Vec2i &ref,
    float w
)
{
    const float normFctor = 255.0f * 255.0f * 3.0f;
    w = 0;

    float ac = 0.0f;
    for (int r = 0; r < windowSize; ++r) {
        uchar *ptrRefMask = mMask[W_BORDER].ptr<uchar>(r + ref[0]);
        uchar *ptrTargetMask = mMask[W_BORDER].ptr<uchar>(r + target[0]);
        uchar *ptrTargetDiscarding = mDiscardings[W_BORDER].ptr<uchar>(r + target[0]);
        uchar *ptrRefDiscarding = mDiscardings[W_BORDER].ptr<uchar>(r + ref[0]);
        cv::Vec3b *ptrTargetColor = mColor[W_BORDER].ptr<cv::Vec3b>(r + target[0]);
        cv::Vec3b *ptrRefColor = mColor[W_BORDER].ptr<cv::Vec3b>(r + ref[0]);
        for (int c = 0; c < windowSize; ++c) {
            if (ptrRefMask[c + ref[1]] == 0 || ptrRefDiscarding[c + ref[1]] == 0
                || ptrTargetDiscarding[c + target[1]] || ptrTargetMask[c + target[1]] == 0) {
                //ac += FLT_MAX * w;
            }
            else {
                cv::Vec3f diff(cv::Vec3f(ptrTargetColor[c + target[1]]) - cv::Vec3f(ptrRefColor[c + ref[1]]));
                ac += diff.dot(diff);
                w++;
            }
        }
    }

    w = 1.0f / w;
    return ac * w / normFctor;
}

float OneLvPixMix::calcConstrCost(
    const cv::Vec2i& target,
    const cv::Vec2i& ref
)
{
    return 0;
    // image gradient distance
    constexpr float normFactor = 255.0f * 255.0f;
    float w = 0;
    int halfWindowSize = windowSize / 2;
    std::vector<float> weightPerGrid;
    float mean = 0;
    for (int r = 0; r < windowSize; r++) {
        for (int c = 0; c < windowSize; c++) {
            int diffR = abs(halfWindowSize - r);
            int diffC = abs(halfWindowSize - c);
            float weight = 2 * std::min(diffR, diffC) + 1;
            weightPerGrid.push_back(weight);
            mean += weight;
        }
    }
    mean = mean / weightPerGrid.size();

    //std::vector<cv::Point> refEdges, targetEdges;

    float cc = 0;
    for (int r = 0; r < windowSize; ++r) {
        uchar* ptrRefMask = mMask[W_BORDER].ptr<uchar>(r + ref[0]);
        uchar* ptrTargetMask = mMask[W_BORDER].ptr<uchar>(r + target[0]);
        uchar* ptrTargetDiscarding = mDiscardings[W_BORDER].ptr<uchar>(r + target[0]);
        uchar* ptrRefDiscarding = mDiscardings[W_BORDER].ptr<uchar>(r + ref[0]);
        uchar* ptrTargetGradient = mGradient[W_BORDER].ptr<uchar>(r + target[0]);
        uchar* ptrRefGradient = mGradient[W_BORDER].ptr<uchar>(r + ref[0]);
        for (int c = 0; c < windowSize; ++c) {
            if (ptrRefMask[c + ref[1]] != 0 && ptrRefDiscarding[c + ref[1]] != 0
                && ptrTargetMask[c + target[1]] != 0 && ptrTargetDiscarding[c + target[1]] != 0) {
                cc += (ptrTargetGradient[c + target[1]] - ptrRefGradient[c + ref[1]]) * (ptrTargetGradient[c + target[1]] - ptrRefGradient[c + ref[1]]) * weightPerGrid.at(r * windowSize + c);

                //if (ptrRefGradient[c + ref[1]] != 0) refEdges.push_back(cv::Point(c, r));
                //if (ptrTargetGradient[c + ref[1]] != 0) targetEdges.push_back(cv::Point(c, r));

                //w++;
                w += weightPerGrid.at(r * windowSize + c);
            }
        }
    }

    if (w == 0) return std::numeric_limits<float>::infinity();
    w = 1.0f / w;

    //float dist = static_cast<float>(windowSize) * sqrt(2);
    //if (refEdges.size() != 0 && targetEdges.size() != 0) {
    //    auto hde = cv::createHausdorffDistanceExtractor();
    //    dist = hde->computeDistance(refEdges, targetEdges);
    //}
    //return dist * w / (static_cast<float>(windowSize) * sqrt(2));

    return cc * w / normFactor;
}

float OneLvPixMix::calcDummyCost(
    const cv::Vec2i &ref
)
{
    uchar* ptrDiscardings = mDiscardings[WO_BORDER].ptr<uchar>(ref[0]);
    if (ptrDiscardings[ref[1]] == 0) return std::numeric_limits<float>::max();    // this isn't in the source area
    else return 0;
}

cv::Mat OneLvPixMix::leftVerticalBoundaryCut(const cv::Mat_<cv::Vec3b>& newPatch, const cv::Vec2i& center)
{
    cv::Mat_<uchar> newPatchMask = cv::Mat_<uchar>(newPatch.rows, newPatch.cols, (uchar)0);
    for (int i = borderSize; i >= -borderSize; i--) {   // â∫Ç©ÇÁè„Ç÷íTçı
        std::vector<float> e_h;

        for (int j = -1; j >= -borderSize; j--) { // âEÇ©ÇÁç∂Ç÷íTçıÅDç∂îºï™ÇæÇØèàóù
            if (center[0] + i < 0 || center[1] + j < 0 || center[0] + i >= mColor[WO_BORDER].rows || center[1] + j >= mColor[WO_BORDER].cols) {
                e_h.push_back(std::numeric_limits<float>::max());
            }
            else {
                float e = cv::norm(mColor[WO_BORDER](center[0] + i, center[1] + j) - newPatch(borderSize + i, borderSize + j));
                e_h.push_back(e);
            }
        }

        auto min_itr = std::min_element(e_h.begin(), e_h.end());
        int min_idx = std::distance(e_h.begin(), min_itr);

        for (int k = min_idx; k >= 0; k--) {
            newPatchMask(borderSize + i, borderSize - (k + 1)) = 1; // newPatchÇ≈ñÑÇﬂÇÈïîï™ÇÕ1Ç∆Ç∑ÇÈ
        }
    }

    return newPatchMask;
}

cv::Mat OneLvPixMix::rightVerticalBoundaryCut(const cv::Mat_<cv::Vec3b>& newPatch, const cv::Vec2i& center)
{
    cv::Mat_<uchar> newPatchMask = cv::Mat_<uchar>(newPatch.rows, newPatch.cols, (uchar)0);
    for (int i = -borderSize; i <= borderSize; i++) {   // è„Ç©ÇÁâ∫Ç÷íTçı
        std::vector<float> e_h;

        for (int j = 1; j <= borderSize; j++) { // ç∂Ç©ÇÁâEÇ÷íTçıÅDâEîºï™ÇæÇØèàóù
            if (center[0] + i < 0 || center[1] + j < 0 || center[0] + i >= mColor[WO_BORDER].rows || center[1] + j >= mColor[WO_BORDER].cols) {
                e_h.push_back(std::numeric_limits<float>::max());
            }
            else {
                float e = cv::norm(mColor[WO_BORDER](center[0] + i, center[1] + j) - newPatch(borderSize + i, borderSize + j));
                e_h.push_back(e);
            }
        }

        auto min_itr = std::min_element(e_h.begin(), e_h.end());
        int min_idx = std::distance(e_h.begin(), min_itr);

        for (int k = min_idx; k >= 0; k--) {
            newPatchMask(borderSize + i, borderSize + (k + 1)) = 1; // newPatchÇ≈ñÑÇﬂÇÈïîï™ÇÕ1Ç∆Ç∑ÇÈ
        }
    }

    return newPatchMask;
}

cv::Mat OneLvPixMix::topHorizontalBoundaryCut(const cv::Mat_<cv::Vec3b>& newPatch, const cv::Vec2i& center)
{
    cv::Mat_<uchar> newPatchMask = cv::Mat_<uchar>(newPatch.rows, newPatch.cols, (uchar)0);
    for (int j = borderSize; j >= -borderSize; j--) {   // âEÇ©ÇÁç∂Ç÷íTçı
        std::vector<float> e_v;

        for (int i = -1; i >= -borderSize; i--) {   // â∫Ç©ÇÁè„Ç÷íTçı
            if (center[0] + i < 0 || center[1] + j < 0 || center[0] + i >= mColor[WO_BORDER].rows || center[1] + j >= mColor[WO_BORDER].cols) {
                e_v.push_back(std::numeric_limits<float>::max());
            }
            else {
                float e = cv::norm(mColor[WO_BORDER](center[0] + i, center[1] + j) - newPatch(borderSize + i, borderSize + j));
                e_v.push_back(e);
            }
        }

        auto min_itr = std::min_element(e_v.begin(), e_v.end());
        int min_idx = std::distance(e_v.begin(), min_itr);

        for (int k = min_idx; k >= 0; k--) {
            newPatchMask(borderSize - (k + 1), borderSize + j) = 1; // newPatchÇ≈ñÑÇﬂÇÈïîï™ÇÕ1Ç∆Ç∑ÇÈ
        }
    }

    return newPatchMask;
}

cv::Mat OneLvPixMix::bottomHorizontalBoundaryCut(const cv::Mat_<cv::Vec3b>& newPatch, const cv::Vec2i& center)
{
    cv::Mat_<uchar> newPatchMask = cv::Mat_<uchar>(newPatch.rows, newPatch.cols, (uchar)0);
    for (int j = -borderSize; j <= borderSize; j++) {   // ç∂Ç©ÇÁâEÇ…íTçı
        std::vector<float> e_v;

        for (int i = 1; i <= borderSize; i++) { // è„Ç©ÇÁâ∫Ç÷íTçı
            if (center[0] + i < 0 || center[1] + j < 0 || center[0] + i >= mColor[WO_BORDER].rows || center[1] + j >= mColor[WO_BORDER].cols) {
                e_v.push_back(std::numeric_limits<float>::max());
            }
            else {
                float e = cv::norm(mColor[WO_BORDER](center[0] + i, center[1] + j) - newPatch(borderSize + i, borderSize + j));
                e_v.push_back(e);
            }
        }

        auto min_itr = std::min_element(e_v.begin(), e_v.end());
        int min_idx = std::distance(e_v.begin(), min_itr);

        for (int k = min_idx; k >= 0; k--) {
            newPatchMask(borderSize + (k + 1), borderSize + j) = 1; // newPatchÇ≈ñÑÇﬂÇÈïîï™ÇÕ1Ç∆Ç∑ÇÈ
        }
    }

    return newPatchMask;
}

void OneLvPixMix::fwdUpdate(
    const float scAlpha,
    const float acAlpha,
    const float ccAlpha,
    const float thDist,
    const int maxRandSearchItr
)
{
#pragma omp parallel for // NOTE: This is not thread-safe
    for (int r = 0; r < mColor[WO_BORDER].rows; ++r) {
        uchar *ptrMask = mMask[WO_BORDER].ptr<uchar>(r);
        cv::Vec2i *ptrPosMap = mPosMap[WO_BORDER].ptr<cv::Vec2i>(r);
        for (int c = 0; c < mColor[WO_BORDER].cols; ++c) {
            if (ptrMask[c] == 0) {
                cv::Vec2i target(r, c);
                cv::Vec2i ref = ptrPosMap[target[1]];
                cv::Vec2i top = target + toUp;
                cv::Vec2i left = target + toLeft;
                if (top[0] < 0) top[0] = 0;
                if (left[1] < 0) left[1] = 0;
                cv::Vec2i topRef = mPosMap[WO_BORDER](top) + toDown;
                cv::Vec2i leftRef = mPosMap[WO_BORDER](left) + toRight;
                if (topRef[0] >= mColor[WO_BORDER].rows) topRef[0] = mPosMap[WO_BORDER](top)[0];
                if (leftRef[1] >= mColor[WO_BORDER].cols) leftRef[1] = mPosMap[WO_BORDER](left)[1];

                // propagate
                float cost = scAlpha * calcSptCost(target, ref, thDist) + acAlpha * calcAppCost(target, ref) + ccAlpha * calcConstrCost(target, ref) + calcDummyCost(ref);
                float costTop = FLT_MAX, costLeft = FLT_MAX;

                if (mMask[WO_BORDER](top) == 0 && mMask[WO_BORDER](topRef) != 0) {
                    costTop = scAlpha * calcSptCost(target, topRef, thDist) + acAlpha * calcAppCost(target, topRef) + ccAlpha * calcConstrCost(target, topRef) + calcDummyCost(topRef);
                }
                if (mMask[WO_BORDER](left) == 0 && mMask[WO_BORDER](leftRef) != 0) {
                    costLeft = scAlpha * calcSptCost(target, leftRef, thDist) + acAlpha * calcAppCost(target, leftRef) + ccAlpha * calcConstrCost(target, leftRef) + calcDummyCost(leftRef);
                }

                if (costTop < cost && costTop < costLeft) {
                    cost = costTop;
                    ptrPosMap[target[1]] = topRef;
                }
                else if (costLeft < cost) {
                    cost = costLeft;
                    ptrPosMap[target[1]] = leftRef;
                }

                // random search
                int itrNum = 0;
                cv::Vec2i refRand;
                float costRand = FLT_MAX;
                do {
                    refRand = getValidRandPos();
                    costRand = scAlpha * calcSptCost(target, refRand, thDist) + acAlpha * calcAppCost(target, refRand) + ccAlpha * calcConstrCost(target, refRand) + calcDummyCost(refRand);
                } while (costRand >= cost && ++itrNum < maxRandSearchItr);

                if (costRand < cost) ptrPosMap[target[1]] = refRand;

                cv::Mat_<cv::Vec3b> newPatch(windowSize, windowSize, cv::Vec3b::all(0));
                for (int i = -borderSize; i <= borderSize; i++) {
                    for (int j = -borderSize; j <= borderSize; j++) {
                        if (target[0] + i < 0 || target[1] + j < 0 || target[0] + i >= mPosMap[WO_BORDER].rows || target[1] + j >= mPosMap[WO_BORDER].cols) continue;
                        if (ptrPosMap[target[1]][0] + i < 0 || ptrPosMap[target[1]][1] + j < 0 || ptrPosMap[target[1]][0] + i >= mPosMap[WO_BORDER].rows || ptrPosMap[target[1]][1] + j >= mPosMap[WO_BORDER].cols) continue;

                        mPosMap[WO_BORDER](target[0] + i, target[1] + j) = ptrPosMap[target[1]] + cv::Vec2i(i, j);
                        newPatch(borderSize + i, borderSize + j) = mColor[WO_BORDER](ptrPosMap[target[1]] + cv::Vec2i(i, j));
                    }
                }

                cv::Mat m = bottomHorizontalBoundaryCut(newPatch, target);
                std::cout << m << std::endl;
            }
        }
    }
}

void OneLvPixMix::bwdUpdate(
    const float scAlpha,
    const float acAlpha,
    const float ccAlpha,
    const float thDist,
    const int maxRandSearchItr
)
{
#pragma omp parallel for // NOTE: This is not thread-safe
    for (int r = mColor[WO_BORDER].rows - 1; r >= 0; --r) {
        uchar *ptrMask = mMask[WO_BORDER].ptr<uchar>(r);
        cv::Vec2i *ptrPosMap = mPosMap[WO_BORDER].ptr<cv::Vec2i>(r);
        for (int c = mColor[WO_BORDER].cols - 1; c >= 0; --c) {
            if (ptrMask[c] == 0) {
                cv::Vec2i target(r, c);
                cv::Vec2i ref = ptrPosMap[target[1]];
                cv::Vec2i bottom = target + toDown;
                cv::Vec2i right = target + toRight;
                if (bottom[0] >= mColor[WO_BORDER].rows) bottom[0] = target[0];
                if (right[1] >= mColor[WO_BORDER].cols) right[1] = target[1];
                cv::Vec2i bottomRef = mPosMap[WO_BORDER](bottom) + toUp;
                cv::Vec2i rightRef = mPosMap[WO_BORDER](right) + toLeft;
                if (bottomRef[0] < 0) bottomRef[0] = 0;
                if (rightRef[1] < 0) rightRef[1] = 0;

                // propagate
                float cost = scAlpha * calcSptCost(target, ref, thDist) + acAlpha * calcAppCost(target, ref) + ccAlpha * calcConstrCost(target, ref) + calcDummyCost(ref);
                float costTop = FLT_MAX, costLeft = FLT_MAX;

                if (mMask[WO_BORDER](bottom) == 0 && mMask[WO_BORDER](bottomRef) != 0) {
                    costTop = scAlpha * calcSptCost(target, bottomRef, thDist) + acAlpha * calcAppCost(target, bottomRef) + ccAlpha * calcConstrCost(target, bottomRef) + calcDummyCost(bottomRef);
                }
                if (mMask[WO_BORDER](right) == 0 && mMask[WO_BORDER](rightRef) != 0) {
                    costLeft = scAlpha * calcSptCost(target, rightRef, thDist) + acAlpha * calcAppCost(target, rightRef) + ccAlpha * calcConstrCost(target, rightRef) + calcDummyCost(rightRef);
                }

                if (costTop < cost && costTop < costLeft) {
                    cost = costTop;
                    ptrPosMap[target[1]] = bottomRef;
                }
                else if (costLeft < cost) {
                    cost = costLeft;
                    ptrPosMap[target[1]] = rightRef;
                }

                // random search
                int itrNum = 0;
                cv::Vec2i refRand;
                float costRand = FLT_MAX;
                do {
                    refRand = getValidRandPos();
                    costRand = scAlpha * calcSptCost(target, refRand, thDist) + acAlpha * calcAppCost(target, refRand) + ccAlpha * calcConstrCost(target, refRand) + calcDummyCost(refRand);
                } while (costRand >= cost && ++itrNum < maxRandSearchItr);

                if (costRand < cost) ptrPosMap[target[1]] = refRand;

                for (int i = -borderSize; i <= borderSize; i++) {
                    for (int j = -borderSize; j <= borderSize; j++) {
                        if (target[0] + i < 0 || target[1] + j < 0 || target[0] + i >= mPosMap[WO_BORDER].rows || target[1] + j >= mPosMap[WO_BORDER].cols) continue;
                        if (ptrPosMap[target[1]][0] + i < 0 || ptrPosMap[target[1]][1] + j < 0 || ptrPosMap[target[1]][0] + i >= mPosMap[WO_BORDER].rows || ptrPosMap[target[1]][1] + j >= mPosMap[WO_BORDER].cols) continue;

                        mPosMap[WO_BORDER](target[0] + i, target[1] + j) = ptrPosMap[target[1]] + cv::Vec2i(i, j);
                    }
                }
            }
        }
    }
}