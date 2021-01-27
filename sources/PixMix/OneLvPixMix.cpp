#include "OneLvPixMix.h"

#include <numeric>

OneLvPixMix::OneLvPixMix()
    : borderSize(7), borderSizePosMap(1), borderSizeColorPatch(2), windowSize(15), windowSizeColorPatch(5), toLeft(0, -1), toRight(0, 1), toUp(-1, 0), toDown(1, 0) // borderSize = windowSize / 2
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
    const cv::Mat_<uchar> &discarding_area
)
{
    std::random_device rnd;
    mt = std::mt19937(rnd());
    cRand = std::uniform_int_distribution<int>(0, color.cols - 1);
    rRand = std::uniform_int_distribution<int>(0, color.rows - 1);

    cv::copyMakeBorder(color, mColor[W_BORDER], borderSize, borderSize, borderSize, borderSize, cv::BORDER_REFLECT);
    cv::copyMakeBorder(mask, mMask[W_BORDER], borderSize, borderSize, borderSize, borderSize, cv::BORDER_REFLECT);
    cv::copyMakeBorder(discarding_area, mDiscardings[W_BORDER], borderSize, borderSize, borderSize, borderSize, cv::BORDER_REFLECT);
    mColor[WO_BORDER] = cv::Mat(mColor[W_BORDER], cv::Rect(borderSize, borderSize, color.cols, color.rows));
    mMask[WO_BORDER] = cv::Mat(mMask[W_BORDER], cv::Rect(borderSize, borderSize, mask.cols, mask.rows));
    mDiscardings[WO_BORDER] = cv::Mat(mDiscardings[W_BORDER], cv::Rect(borderSize, borderSize, discarding_area.cols, discarding_area.rows));

    mPosMap[WO_BORDER] = cv::Mat_<cv::Vec2i>(mColor[WO_BORDER].size());
    for (int r = 0; r < mPosMap[WO_BORDER].rows; ++r) {
        for (int c = 0; c < mPosMap[WO_BORDER].cols; ++c) {
            if (mMask[WO_BORDER](r, c) == 0) mPosMap[WO_BORDER](r, c) = getValidRandPos();
            else mPosMap[WO_BORDER](r, c) = cv::Vec2i(r, c);
        }
    }
    cv::copyMakeBorder(mPosMap[WO_BORDER], mPosMap[W_BORDER], borderSizePosMap, borderSizePosMap, borderSizePosMap, borderSizePosMap, cv::BORDER_REFLECT);
    mPosMap[WO_BORDER] = cv::Mat(mPosMap[W_BORDER], cv::Rect(borderSizePosMap, borderSizePosMap, color.cols, color.rows));
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

    // appearance costのみで初期化
    bwdUpdate(0.0f, 1.0f, 0.0f, thDist, maxRandSearchItr);

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
                || (ptrTargetDiscarding[c + target[1]] == 0 && ptrTargetMask[c + target[1]] != 0)) { // 除外エリアでマスクではないところ
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
    /*
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
    */
}

float OneLvPixMix::calcDummyCost(
    const cv::Vec2i &ref
)
{
    uchar* ptrDiscardings = mDiscardings[WO_BORDER].ptr<uchar>(ref[0]);
    if (ptrDiscardings[ref[1]] == 0) return std::numeric_limits<float>::max();    // this isn't in the source area
    else return 0;
}

float OneLvPixMix::calcVerticalBoundaryError(const cv::Mat_<cv::Vec3b>& newPatch, const cv::Mat_<uchar>& maskPatch, const cv::Mat_<uchar>& discardPatch, const cv::Vec2i& colorRef, const cv::Vec2i& patchRef, const int minCol, const int maxCol, const BOUNDARY_POSITION pos, std::vector<cv::Vec2i>& path)
{
    float error = cv::norm(mColor[WO_BORDER](colorRef) - newPatch(patchRef));
    path.push_back(patchRef);

    int colori = pos == BOUNDARY_POSITION::LEFT ? colorRef[0] + 1 : colorRef[0] - 1;
    int patchi = pos == BOUNDARY_POSITION::LEFT ? patchRef[0] + 1 : patchRef[0] - 1;

    if (colori < 0 || colori >= mColor[WO_BORDER].rows
        || patchi < 0 || patchi >= newPatch.rows) {
        return error;
    }

    float leftBottomError = std::numeric_limits<float>::max(), bottomError = std::numeric_limits<float>::max(), rightBottomError = std::numeric_limits<float>::max();
    std::vector<cv::Vec2i> path1, path2, path3;
    if (patchRef[1] - 1 >= minCol && patchRef[1] - 1 <= maxCol
        && colorRef[1] - 1 >= 0 && colorRef[1] - 1 < mColor[WO_BORDER].cols
        && patchRef[1] - 1 >= 0 && patchRef[1] - 1 < newPatch.cols) leftBottomError = calcVerticalBoundaryError(newPatch, maskPatch, discardPatch, cv::Vec2i(colori, colorRef[1] - 1), cv::Vec2i(patchi, patchRef[1] - 1), minCol, maxCol, pos, path1);
    bottomError = calcVerticalBoundaryError(newPatch, maskPatch, discardPatch, cv::Vec2i(colori, colorRef[1]), cv::Vec2i(patchi, patchRef[1]), minCol, maxCol, pos, path2);
    if (patchRef[1] + 1 >= minCol && patchRef[1] + 1 <= maxCol
        && colorRef[1] + 1 >= 0 && colorRef[1] + 1 < mColor[WO_BORDER].cols
        && patchRef[1] + 1 >= 0 && patchRef[1] + 1 < newPatch.cols) rightBottomError = calcVerticalBoundaryError(newPatch, maskPatch, discardPatch, cv::Vec2i(colori, colorRef[1] + 1), cv::Vec2i(patchi, patchRef[1] + 1), minCol, maxCol, pos, path3);

    float minError = std::min(leftBottomError, std::min(bottomError, rightBottomError));
    if (minError == std::numeric_limits<float>::max()) {
        return error;   // 計算されなかった場合はそのまま返す
    }

    if (minError == leftBottomError) {
        path.insert(path.end(), path1.begin(), path1.end());
    }
    else if (minError == bottomError) {
        path.insert(path.end(), path2.begin(), path2.end());
    }
    else {
        path.insert(path.end(), path3.begin(), path3.end());
    }

    return error + minError;
}

float OneLvPixMix::calcHorizontalBoundaryError(const cv::Mat_<cv::Vec3b>& newPatch, const cv::Mat_<uchar>& maskPatch, const cv::Mat_<uchar>& discardPatch, const cv::Vec2i& colorRef, const cv::Vec2i& patchRef, const int minRow, const int maxRow, const BOUNDARY_POSITION pos, std::vector<cv::Vec2i>& path)
{
    float error = cv::norm(mColor[WO_BORDER](colorRef) - newPatch(patchRef));
    path.push_back(patchRef);

    int colorj = pos == BOUNDARY_POSITION::TOP ? colorRef[1] + 1 : colorRef[1] - 1;
    int patchj = pos == BOUNDARY_POSITION::TOP ? patchRef[1] + 1 : patchRef[1] - 1;

    if (colorj < 0 || colorj >= mColor[WO_BORDER].cols
        || patchj < 0 || patchj >= newPatch.cols) {
        return error;
    }

    float rightTopError = std::numeric_limits<float>::max(), rightError = std::numeric_limits<float>::max(), rightBottomError = std::numeric_limits<float>::max();
    std::vector<cv::Vec2i> path1, path2, path3;
    if (patchRef[0] - 1 >= minRow && patchRef[0] - 1 <= maxRow
        && colorRef[0] - 1 >= 0 && colorRef[0] - 1 < mColor[WO_BORDER].rows
        && patchRef[0] - 1 >= 0 && patchRef[0] - 1 < newPatch.rows) rightTopError = calcHorizontalBoundaryError(newPatch, maskPatch, discardPatch, cv::Vec2i(colorRef[0] - 1, colorj), cv::Vec2i(patchRef[0] - 1, patchj), minRow, maxRow, pos, path1);
    rightError = calcHorizontalBoundaryError(newPatch, maskPatch, discardPatch, cv::Vec2i(colorRef[0], colorj), cv::Vec2i(patchRef[0], patchj), minRow, maxRow, pos, path2);
    if (patchRef[0] + 1 >= minRow && patchRef[0] + 1 <= maxRow
        && colorRef[0] + 1 >= 0 && colorRef[0] + 1 < mColor[WO_BORDER].rows
        && patchRef[0] + 1 >= 0 && colorRef[0] + 1 < newPatch.rows) rightBottomError = calcHorizontalBoundaryError(newPatch, maskPatch, discardPatch, cv::Vec2i(colorRef[0] + 1, colorj), cv::Vec2i(patchRef[0] + 1, patchj), minRow, maxRow, pos, path3);

    float minError = std::min(rightTopError, std::min(rightError, rightBottomError));
    if (minError == std::numeric_limits<float>::max()) {
        return error;   // 計算されなかった場合はそのまま返す
    }

    if (minError == rightTopError) {
        path.insert(path.end(), path1.begin(), path1.end());
    }
    else if (minError == rightError) {
        path.insert(path.end(), path2.begin(), path2.end());
    }
    else {
        path.insert(path.end(), path3.begin(), path3.end());
    }

    return error + minError;
}

cv::Mat OneLvPixMix::leftVerticalBoundaryCut(const cv::Mat_<cv::Vec3b>& newPatch, const cv::Mat_<uchar>& maskPatch, const cv::Mat_<uchar>& discardPatch, const cv::Vec2i& center)
{
    cv::Mat_<uchar> newPatchMask = cv::Mat_<uchar>(newPatch.rows, newPatch.cols, (uchar)0);
    //newPatchMask(borderSizeColorPatch, borderSizeColorPatch) = 1;
    for (int i = 0; i <= borderSizeColorPatch; i++) {
        for (int j = 0; j <= borderSizeColorPatch; j++) {
            if (maskPatch(borderSizeColorPatch + i, borderSizeColorPatch + j) != 0
                && discardPatch(borderSizeColorPatch + i, borderSizeColorPatch + j) != 0) newPatchMask(borderSizeColorPatch + i, borderSizeColorPatch + j) = 1; // 右下部分
        }
    }

    for (int i = borderSizeColorPatch; i >= -borderSizeColorPatch; i--) {   // 下から上へ探索
        std::vector<float> e_h;

        for (int j = 0; j >= -borderSizeColorPatch; j--) { // 右から左へ探索．左半分だけ処理
            if (center[0] + i < 0 || center[1] + j < 0 || center[0] + i >= mColor[WO_BORDER].rows || center[1] + j >= mColor[WO_BORDER].cols
                || maskPatch(borderSizeColorPatch + i, borderSizeColorPatch + j) == 0   // マスク領域
                || discardPatch(borderSizeColorPatch + i, borderSizeColorPatch + j) == 0) { // 除外エリア
                e_h.push_back(std::numeric_limits<float>::max());
            }
            else {
                float e = cv::norm(mColor[WO_BORDER](center[0] + i, center[1] + j) - newPatch(borderSizeColorPatch + i, borderSizeColorPatch + j));
                e_h.push_back(e);
            }
        }

        auto min_itr = std::min_element(e_h.begin(), e_h.end());
        int min_idx = std::distance(e_h.begin(), min_itr);

        for (int k = min_idx; k >= 0; k--) {
            if (maskPatch(borderSizeColorPatch + i, borderSizeColorPatch - k) != 0
                && discardPatch(borderSizeColorPatch + i, borderSizeColorPatch - k) != 0) newPatchMask(borderSizeColorPatch + i, borderSizeColorPatch - k) = 1; // newPatchで埋める部分は1とする
        }
    }

    return newPatchMask;
}

cv::Mat OneLvPixMix::rightVerticalBoundaryCut(const cv::Mat_<cv::Vec3b>& newPatch, const cv::Mat_<uchar>& maskPatch, const cv::Mat_<uchar>& discardPatch, const cv::Vec2i& center)
{
    cv::Mat_<uchar> newPatchMask = cv::Mat_<uchar>(newPatch.rows, newPatch.cols, (uchar)0);
    //newPatchMask(borderSizeColorPatch, borderSizeColorPatch) = 1;
    for (int i = -borderSizeColorPatch; i <= 0; i++) {
        for (int j = -borderSizeColorPatch; j <= 0; j++) {
            if (maskPatch(borderSizeColorPatch + i, borderSizeColorPatch + j) != 0
                && discardPatch(borderSizeColorPatch + i, borderSizeColorPatch + j) != 0) newPatchMask(borderSizeColorPatch + i, borderSizeColorPatch + j) = 1;   // 左上
        }
    }

    for (int i = -borderSizeColorPatch; i <= borderSizeColorPatch; i++) {   // 上から下へ探索
        std::vector<float> e_h;

        for (int j = 0; j <= borderSizeColorPatch; j++) { // 左から右へ探索．右半分だけ処理
            if (center[0] + i < 0 || center[1] + j < 0 || center[0] + i >= mColor[WO_BORDER].rows || center[1] + j >= mColor[WO_BORDER].cols
                || maskPatch(borderSizeColorPatch + i, borderSizeColorPatch + j) == 0   // マスク領域
                || discardPatch(borderSizeColorPatch + i, borderSizeColorPatch + j) == 0) {
                e_h.push_back(std::numeric_limits<float>::max());
            }
            else {
                float e = cv::norm(mColor[WO_BORDER](center[0] + i, center[1] + j) - newPatch(borderSizeColorPatch + i, borderSizeColorPatch + j));
                e_h.push_back(e);
            }
        }

        auto min_itr = std::min_element(e_h.begin(), e_h.end());
        int min_idx = std::distance(e_h.begin(), min_itr);

        for (int k = min_idx; k >= 0; k--) {
            if (maskPatch(borderSizeColorPatch + i, borderSizeColorPatch + k) != 0
                && discardPatch(borderSizeColorPatch + i, borderSizeColorPatch + k) != 0) newPatchMask(borderSizeColorPatch + i, borderSizeColorPatch + k) = 1; // newPatchで埋める部分は1とする
        }
    }

    return newPatchMask;
}

cv::Mat OneLvPixMix::TopLeftBoundaryCut(const cv::Mat_<cv::Vec3b>& newPatch, const cv::Mat_<uchar>& maskPatch, const cv::Mat_<uchar>& discardPatch, const cv::Vec2i& center)
{
    cv::Mat_<uchar> newPatchMask = cv::Mat_<uchar>(newPatch.rows, newPatch.cols, (uchar)0);
    //newPatchMask(borderSizeColorPatch, borderSizeColorPatch) = 1;
    for (int i = borderSizeColorPatch; i < newPatchMask.rows; i++) {
        for (int j = borderSizeColorPatch; j < newPatchMask.cols; j++) {
            if (maskPatch(i, j) != 0
                && discardPatch(i, j) != 0) newPatchMask(i, j) = 1; // 右下部分
        }
    }

    std::vector<std::vector<cv::Vec2i>> paths1, paths2;
    std::vector<float> errors1, errors2;
    for (int i = -borderSizeColorPatch; i <= 0; i++) {    // 上側
        std::vector<cv::Vec2i> path;
        float e = calcHorizontalBoundaryError(newPatch, maskPatch, discardPatch, cv::Vec2i(center[0] + i, center[1] + i), cv::Vec2i(borderSizeColorPatch + i, borderSizeColorPatch + i), 0, borderSizeColorPatch, BOUNDARY_POSITION::TOP, path);
        paths1.push_back(path);
        errors1.push_back(e);
    }

    for (int j = -borderSizeColorPatch; j <= 0; j++) {    // 左側
        std::vector<cv::Vec2i> path;
        float e = calcVerticalBoundaryError(newPatch, maskPatch, discardPatch, cv::Vec2i(center[0] + j, center[1] + j), cv::Vec2i(borderSizeColorPatch + j, borderSizeColorPatch + j), 0, borderSizeColorPatch, BOUNDARY_POSITION::LEFT, path);
        paths2.push_back(path);
        errors2.push_back(e);
    }

    float min = std::numeric_limits<float>::max();
    int min_idx = 0;
    for (int i = 0, iend = paths1.size(); i < iend; i++) {
        float e = errors1[i] + errors2[i] - cv::norm(mColor[WO_BORDER](center[0] - borderSizeColorPatch + paths1[i][0][0], center[1] - borderSizeColorPatch + paths1[i][0][1]) - newPatch(paths1[i][0]));
        if (e < min) {
            min = e;
            min_idx = i;
        }
    }

    std::vector<cv::Vec2i> path_h = paths1[min_idx];
    std::vector<cv::Vec2i> path_v = paths2[min_idx];

    for (int i = 0, iend = path_h.size(); i < iend; i++) {
        int c = path_h[i][1];
        for (int k = path_h[i][0]; k <= borderSizeColorPatch; k++) {  // 各列で下方向に探索
            if (newPatchMask(k, c) == 1) break;
            if (maskPatch(k, c) != 0 && discardPatch(k, c) != 0) newPatchMask(k, c) = 1;

            bool isEnd = std::find_if(path_v.begin(), path_v.end(), [&](const auto& v) {
                return v[0] == k && v[1] == c;
            }) != path_v.end();
            if (isEnd) break;
        }
    }
    for (int i = 0, iend = path_v.size(); i < iend; i++) {
        int r = path_v[i][0];
        for (int k = path_v[i][1]; k <= borderSizeColorPatch; k++) {    // 各行で右方向に探索
            if (newPatchMask(r, k) == 1) break;
            if (maskPatch(r, k) != 0 && discardPatch(r, k) != 0) newPatchMask(r, k) = 1;

            bool isEnd = std::find_if(path_h.begin(), path_h.end(), [&](const auto& h) {
                return h[0] == r && h[1] == k;
            }) != path_h.end();
            if (isEnd) break;
        }
    }

    //std::cout << newPatchMask << std::endl;
    return newPatchMask;
}

cv::Mat OneLvPixMix::TopRightBoundaryCut(const cv::Mat_<cv::Vec3b>& newPatch, const cv::Mat_<uchar>& maskPatch, const cv::Mat_<uchar>& discardPatch, const cv::Vec2i& center)
{
    cv::Mat_<uchar> newPatchMask = cv::Mat_<uchar>(newPatch.rows, newPatch.cols, (uchar)0);
    //newPatchMask(borderSizeColorPatch, borderSizeColorPatch) = 1;
    for (int i = borderSizeColorPatch; i < windowSizeColorPatch; i++) {
        for (int j = 0; j <= borderSizeColorPatch; j++) {
            if (maskPatch(i, j) != 0
                && discardPatch(i, j) != 0) newPatchMask(i, j) = 1; // 左下部分
        }
    }

    std::vector<std::vector<cv::Vec2i>> paths1, paths2;
    std::vector<float> errors1, errors2;
    for (int i = -borderSizeColorPatch; i <= 0; i++) {    // 上側
        std::vector<cv::Vec2i> path;
        float e = calcHorizontalBoundaryError(newPatch, maskPatch, discardPatch, cv::Vec2i(center[0] + i, center[1] - i), cv::Vec2i(borderSizeColorPatch + i, borderSizeColorPatch - i), 0, borderSizeColorPatch, BOUNDARY_POSITION::BOTTOM, path);
        paths1.push_back(path);
        errors1.push_back(e);
    }

    for (int j = borderSizeColorPatch; j >= 0; j--) { // 右側
        std::vector<cv::Vec2i> path;
        float e = calcVerticalBoundaryError(newPatch, maskPatch, discardPatch, cv::Vec2i(center[0] - j, center[1] + j), cv::Vec2i(borderSizeColorPatch - j, borderSizeColorPatch + j), borderSizeColorPatch, newPatch.cols - 1, BOUNDARY_POSITION::LEFT, path);
        paths2.push_back(path);
        errors2.push_back(e);
    }

    float min = std::numeric_limits<float>::max();
    int min_idx = 0;
    for (int i = 0, iend = paths1.size(); i < iend; i++) {
        float e = errors1[i] + errors2[i] - cv::norm(mColor[WO_BORDER](center[0] - borderSizeColorPatch + paths1[i][0][0], center[1] - borderSizeColorPatch + paths1[i][0][1]) - newPatch(paths1[i][0]));
        if (e < min) {
            min = e;
            min_idx = i;
        }
    }

    std::vector<cv::Vec2i> path_h = paths1[min_idx];
    std::vector<cv::Vec2i> path_v = paths2[min_idx];

    for (int i = 0, iend = path_h.size(); i < iend; i++) {
        int c = path_h[i][1];
        for (int k = path_h[i][0]; k <= borderSizeColorPatch; k++) {  // 各列で下方向に探索
            if (newPatchMask(k, c) == 1) break;
            if (maskPatch(k, c) != 0 && discardPatch(k, c) != 0) newPatchMask(k, c) = 1;

            bool isEnd = std::find_if(path_v.begin(), path_v.end(), [&](const auto& v) {
                return v[0] == k && v[1] == c;
            }) != path_v.end();
            if (isEnd) break;
        }
    }
    for (int i = 0, iend = path_v.size(); i < iend; i++) {
        int r = path_v[i][0];
        for (int k = path_v[i][1]; k >= borderSizeColorPatch; k--) {  // 各行で左方向に探索
            if (newPatchMask(r, k) == 1) break;
            if (maskPatch(r, k) != 0 && discardPatch(r, k) != 0) newPatchMask(r, k) = 1;

            bool isEnd = std::find_if(path_h.begin(), path_h.end(), [&](const auto& h) {
                return h[0] == r && h[1] == k;
            }) != path_h.end();
            if (isEnd) break;
        }
    }

    //std::cout << newPatchMask << std::endl;
    return newPatchMask;
}

cv::Mat OneLvPixMix::BottomRightBoundaryCut(const cv::Mat_<cv::Vec3b>& newPatch, const cv::Mat_<uchar>& maskPatch, const cv::Mat_<uchar>& discardPatch, const cv::Vec2i& center)
{
    cv::Mat_<uchar> newPatchMask = cv::Mat_<uchar>(newPatch.rows, newPatch.cols, (uchar)0);
    //newPatchMask(borderSizeColorPatch, borderSizeColorPatch) = 1;
    for (int i = 0; i <= borderSizeColorPatch; i++) {
        for (int j = 0; j <= borderSizeColorPatch; j++) {
            if (maskPatch(i, j) != 0
                && discardPatch(i, j) != 0) newPatchMask(i, j) = 1; // 左上部分
        }
    }

    std::vector<std::vector<cv::Vec2i>> paths1, paths2;
    std::vector<float> errors1, errors2;
    for (int i = borderSizeColorPatch; i >= 0; i--) { // 下側
        std::vector<cv::Vec2i> path;
        float e = calcHorizontalBoundaryError(newPatch, maskPatch, discardPatch, cv::Vec2i(center[0] + i, center[1] + i), cv::Vec2i(borderSizeColorPatch + i, borderSizeColorPatch + i), borderSizeColorPatch, newPatch.rows - 1, BOUNDARY_POSITION::BOTTOM, path);
        paths1.push_back(path);
        errors1.push_back(e);
    }

    for (int j = borderSizeColorPatch; j >= 0; j--) { // 右側
        std::vector<cv::Vec2i> path;
        float e = calcVerticalBoundaryError(newPatch, maskPatch, discardPatch, cv::Vec2i(center[0] + j, center[1] + j), cv::Vec2i(borderSizeColorPatch + j, borderSizeColorPatch + j), borderSizeColorPatch, newPatch.cols - 1, BOUNDARY_POSITION::RIGHT, path);
        paths2.push_back(path);
        errors2.push_back(e);
    }

    float min = std::numeric_limits<float>::max();
    int min_idx = 0;
    for (int i = 0, iend = paths1.size(); i < iend; i++) {
        float e = errors1[i] + errors2[i] - cv::norm(mColor[WO_BORDER](center[0] - borderSizeColorPatch + paths1[i][0][0], center[1] - borderSizeColorPatch + paths1[i][0][1]) - newPatch(paths1[i][0]));
        if (e < min) {
            min = e;
            min_idx = i;
        }
    }

    std::vector<cv::Vec2i> path_h = paths1[min_idx];
    std::vector<cv::Vec2i> path_v = paths2[min_idx];

    //for (const auto& p : path_h) {
    //    std::cout << p << " ";
    //}std::cout << std::endl;
    //for (const auto& p : path_v) {
    //    std::cout << p << " ";
    //}std::cout << std::endl;

    for (int i = 0, iend = path_h.size(); i < iend; i++) {
        int c = path_h[i][1];
        for (int k = path_h[i][0]; k >= borderSizeColorPatch; k--) {  // 各列で上方向に探索
            if (newPatchMask(k, c) == 1) break;
            if (maskPatch(k, c) != 0 && discardPatch(k, c) != 0) newPatchMask(k, c) = 1;

            bool isEnd = std::find_if(path_v.begin(), path_v.end(), [&](const auto& v) {
                return v[0] == k && v[1] == c;
            }) != path_v.end();
            if (isEnd) break;
        }
    }
    for (int i = 0, iend = path_v.size(); i < iend; i++) {
        int r = path_v[i][0];
        for (int k = path_v[i][1]; k >= borderSizeColorPatch; k--) {  // 各行で左方向に探索
            if (newPatchMask(r, k) == 1) break;
            if (maskPatch(r, k) != 0 && discardPatch(r, k) != 0) newPatchMask(r, k) = 1;

            bool isEnd = std::find_if(path_h.begin(), path_h.end(), [&](const auto& h) {
                return h[0] == r && h[1] == k;
            }) != path_h.end();
            if (isEnd) break;
        }
    }

    //std::cout << newPatchMask << std::endl;
    return newPatchMask;
}

cv::Mat OneLvPixMix::topHorizontalBoundaryCut(const cv::Mat_<cv::Vec3b>& newPatch, const cv::Mat_<uchar>& maskPatch, const cv::Mat_<uchar>& discardPatch, const cv::Vec2i& center)
{
    cv::Mat_<uchar> newPatchMask = cv::Mat_<uchar>(newPatch.rows, newPatch.cols, (uchar)0);
    //newPatchMask(borderSizeColorPatch, borderSizeColorPatch) = 1;
    for (int i = 0; i <= borderSizeColorPatch; i++) {
        for (int j = 0; j <= borderSizeColorPatch; j++) {
            if (maskPatch(borderSizeColorPatch + i, borderSizeColorPatch + j) != 0
                && discardPatch(borderSizeColorPatch + i, borderSizeColorPatch + j) != 0) newPatchMask(borderSizeColorPatch + i, borderSizeColorPatch + j) = 1; // 右下部分
        }
    }

    for (int j = borderSizeColorPatch; j >= -borderSizeColorPatch; j--) {   // 右から左へ探索
        std::vector<float> e_v;

        for (int i = 0; i >= -borderSizeColorPatch; i--) {   // 下から上へ探索
            if (center[0] + i < 0 || center[1] + j < 0 || center[0] + i >= mColor[WO_BORDER].rows || center[1] + j >= mColor[WO_BORDER].cols
                || maskPatch(borderSizeColorPatch + i, borderSizeColorPatch + j) == 0   // マスク領域
                || discardPatch(borderSizeColorPatch + i, borderSizeColorPatch + j) == 0) {
                e_v.push_back(std::numeric_limits<float>::max());
            }
            else {
                float e = cv::norm(mColor[WO_BORDER](center[0] + i, center[1] + j) - newPatch(borderSizeColorPatch + i, borderSizeColorPatch + j));
                e_v.push_back(e);
            }
        }

        auto min_itr = std::min_element(e_v.begin(), e_v.end());
        int min_idx = std::distance(e_v.begin(), min_itr);

        for (int k = min_idx; k >= 0; k--) {
            if (maskPatch(borderSizeColorPatch - k, borderSizeColorPatch + j) != 0
                && discardPatch(borderSizeColorPatch - k, borderSizeColorPatch + j) != 0) newPatchMask(borderSizeColorPatch - k, borderSizeColorPatch + j) = 1; // newPatchで埋める部分は1とする
        }
    }

    return newPatchMask;
}

cv::Mat OneLvPixMix::bottomHorizontalBoundaryCut(const cv::Mat_<cv::Vec3b>& newPatch, const cv::Mat_<uchar>& maskPatch, const cv::Mat_<uchar>& discardPatch, const cv::Vec2i& center)
{
    cv::Mat_<uchar> newPatchMask = cv::Mat_<uchar>(newPatch.rows, newPatch.cols, (uchar)0);
    //newPatchMask(borderSizeColorPatch, borderSizeColorPatch) = 1;
    for (int i = -borderSizeColorPatch; i <= 0; i++) {
        for (int j = -borderSizeColorPatch; j <= 0; j++) {
            if (maskPatch(borderSizeColorPatch + i, borderSizeColorPatch + j) != 0
                && discardPatch(borderSizeColorPatch + i, borderSizeColorPatch + j) != 0) newPatchMask(borderSizeColorPatch + i, borderSizeColorPatch + j) = 1;   // 左上
        }
    }

    for (int j = -borderSizeColorPatch; j <= borderSizeColorPatch; j++) {   // 左から右に探索
        std::vector<float> e_v;

        for (int i = 0; i <= borderSizeColorPatch; i++) { // 上から下へ探索
            if (center[0] + i < 0 || center[1] + j < 0 || center[0] + i >= mColor[WO_BORDER].rows || center[1] + j >= mColor[WO_BORDER].cols
                || maskPatch(borderSizeColorPatch + i, borderSizeColorPatch + j) == 0   // マスク領域
                || discardPatch(borderSizeColorPatch + i, borderSizeColorPatch + j) == 0) {
                e_v.push_back(std::numeric_limits<float>::max());
            }
            else {
                float e = cv::norm(mColor[WO_BORDER](center[0] + i, center[1] + j) - newPatch(borderSizeColorPatch + i, borderSizeColorPatch + j));
                e_v.push_back(e);
            }
        }

        auto min_itr = std::min_element(e_v.begin(), e_v.end());
        int min_idx = std::distance(e_v.begin(), min_itr);

        for (int k = min_idx; k >= 0; k--) {
            if (maskPatch(borderSizeColorPatch + k, borderSizeColorPatch + j) != 0
                && discardPatch(borderSizeColorPatch + k, borderSizeColorPatch + j) != 0) newPatchMask(borderSizeColorPatch + k, borderSizeColorPatch + j) = 1; // newPatchで埋める部分は1とする
        }
    }

    return newPatchMask;
}

void OneLvPixMix::fwdUpdate(
    const float scAlpha,
    const float acAlpha,
    const float ccAlpha,
    const float thDist,
    const int maxRandSearchItr,
    const bool pixelWise
)
{
#pragma omp parallel for // NOTE: This is not thread-safe
    for (int r = 0; r < mColor[WO_BORDER].rows; ++r) {
        uchar *ptrMask = mMask[WO_BORDER].ptr<uchar>(r);
        cv::Vec2i *ptrPosMap = mPosMap[WO_BORDER].ptr<cv::Vec2i>(r);
        //for (int c = 0; c < mColor[WO_BORDER].cols; ++c) {
        for (int c = mColor[WO_BORDER].cols - 1; c >= 0; --c) {
            if (ptrMask[c] == 0) {
                cv::Vec2i target(r, c);
                cv::Vec2i ref = ptrPosMap[target[1]];
                cv::Vec2i top = target + toUp;
                //cv::Vec2i left = target + toLeft;
                cv::Vec2i right = target + toRight;
                if (top[0] < 0) top[0] = 0;
                //if (left[1] < 0) left[1] = 0;
                if (right[1] >= mColor[WO_BORDER].cols) right[1] = target[1];
                cv::Vec2i topRef = mPosMap[WO_BORDER](top) + toDown;
                //cv::Vec2i leftRef = mPosMap[WO_BORDER](left) + toRight;
                cv::Vec2i rightRef = mPosMap[WO_BORDER](right) + toLeft;
                if (topRef[0] >= mColor[WO_BORDER].rows) topRef[0] = mPosMap[WO_BORDER](top)[0];
                //if (leftRef[1] >= mColor[WO_BORDER].cols) leftRef[1] = mPosMap[WO_BORDER](left)[1];
                if (rightRef[1] < 0) rightRef[1] = 0;

                // propagate
                //float cost = scAlpha * calcSptCost(target, ref, thDist) + acAlpha * calcAppCost(target, ref) + ccAlpha * calcConstrCost(target, ref) + calcDummyCost(ref);
                float cost = scAlpha * calcSptCost(target, ref, thDist) + acAlpha * calcAppCost(target, ref) + ccAlpha * calcConstrCost(target, ref) + calcDummyCost(ref);
                cost = isfinite(cost) ? cost : FLT_MAX;
                float costTop = FLT_MAX, costLeft = FLT_MAX;

                if (mMask[WO_BORDER](top) == 0 && mMask[WO_BORDER](topRef) != 0) {
                    costTop = scAlpha * calcSptCost(target, topRef, thDist) + acAlpha * calcAppCost(target, topRef) + ccAlpha * calcConstrCost(target, topRef) + calcDummyCost(topRef);
                    costTop = isfinite(costTop) ? costTop : FLT_MAX;
                }
                //if (mMask[WO_BORDER](left) == 0 && mMask[WO_BORDER](leftRef) != 0) {
                if (mMask[WO_BORDER](right) == 0 && mMask[WO_BORDER](rightRef) != 0) {
                    //costLeft = scAlpha * calcSptCost(target, leftRef, thDist) + acAlpha * calcAppCost(target, leftRef) + ccAlpha * calcConstrCost(target, leftRef) + calcDummyCost(leftRef);
                    costLeft = scAlpha * calcSptCost(target, rightRef, thDist) + acAlpha * calcAppCost(target, rightRef) + ccAlpha * calcConstrCost(target, rightRef) + calcDummyCost(rightRef);
                    costLeft = isfinite(costLeft) ? costLeft : FLT_MAX;
                }

                if (costTop < cost && costTop < costLeft) {
                    cost = costTop;
                    ptrPosMap[target[1]] = topRef;
                }
                else if (costLeft < cost) {
                    cost = costLeft;
                    //ptrPosMap[target[1]] = leftRef;
                    ptrPosMap[target[1]] = rightRef;
                }

                // random search
                int itrNum = 0;
                cv::Vec2i refRand;
                float costRand = FLT_MAX;
                //do {
                //    refRand = getValidRandPos();
                //    costRand = scAlpha * calcSptCost(target, refRand, thDist) + acAlpha * calcAppCost(target, refRand) + ccAlpha * calcConstrCost(target, refRand) + calcDummyCost(refRand);
                //} while ((std::isinf(cost) && std::isinf(costRand)) || (!std::isinf(cost) && !std::isinf(costRand) && costRand >= cost/* && ++itrNum < maxRandSearchItr*/));

                while (!isfinite(costRand)
                    || (costRand >= cost && itrNum < maxRandSearchItr)) {
                    refRand = getValidRandPos();

                    costRand = scAlpha * calcSptCost(target, refRand, thDist) + acAlpha * calcAppCost(target, refRand) + ccAlpha * calcConstrCost(target, refRand) + calcDummyCost(refRand);
                    costRand = isfinite(costRand) ? costRand : FLT_MAX;

                    ++itrNum;
                }

                if (!isfinite(cost) || costRand < cost) ptrPosMap[target[1]] = refRand;

                if (pixelWise) continue;

                cv::Mat_<cv::Vec3b> newPatch(windowSizeColorPatch, windowSizeColorPatch, cv::Vec3b::all(0));
                cv::Mat_<uchar> maskPatch(windowSizeColorPatch, windowSizeColorPatch, (uchar)0), discardPatch(windowSizeColorPatch, windowSizeColorPatch, (uchar)0);
                for (int i = -borderSizeColorPatch; i <= borderSizeColorPatch; i++) {
                    for (int j = -borderSizeColorPatch; j <= borderSizeColorPatch; j++) {
                        cv::Vec2i srcPos = ptrPosMap[target[1]] + cv::Vec2i(i, j);
                        cv::Vec2i targetPos = cv::Vec2i(target[0] + i, target[1] + j);

                        if (targetPos[0] < 0 || targetPos[1] < 0 || targetPos[0] >= mPosMap[WO_BORDER].rows || targetPos[1] >= mPosMap[WO_BORDER].cols) continue;
                        if (srcPos[0] < 0 || srcPos[1] < 0 || srcPos[0] >= mPosMap[WO_BORDER].rows || srcPos[1] >= mPosMap[WO_BORDER].cols) continue;

                        newPatch(borderSizeColorPatch + i, borderSizeColorPatch + j) = mColor[WO_BORDER](srcPos);
                        maskPatch(borderSizeColorPatch + i, borderSizeColorPatch + j) = mMask[WO_BORDER](srcPos);
                        discardPatch(borderSizeColorPatch + i, borderSizeColorPatch + j) = mDiscardings[WO_BORDER](srcPos);
                    }
                }

                //cv::Mat_<uchar> newPatchMask = TopLeftBoundaryCut(newPatch, maskPatch, discardPatch, target);
                cv::Mat_<uchar> newPatchMask = TopRightBoundaryCut(newPatch, maskPatch, discardPatch, target);

                for (int i = -borderSizeColorPatch; i <= borderSizeColorPatch; i++) {
                    for (int j = -borderSizeColorPatch; j <= borderSizeColorPatch; j++) {
                        if (target[0] + i < 0 || target[1] + j < 0 || target[0] + i >= mPosMap[WO_BORDER].rows || target[1] + j >= mPosMap[WO_BORDER].cols) continue;
                        if (ptrPosMap[target[1]][0] + i < 0 || ptrPosMap[target[1]][1] + j < 0 || ptrPosMap[target[1]][0] + i >= mPosMap[WO_BORDER].rows || ptrPosMap[target[1]][1] + j >= mPosMap[WO_BORDER].cols) continue;

                        if (newPatchMask(borderSizeColorPatch + i, borderSizeColorPatch + j) != 0) {
                            mPosMap[WO_BORDER](target[0] + i, target[1] + j) = ptrPosMap[target[1]] + cv::Vec2i(i, j);  // texture quilting
                        }
                    }
                }
            }
        }
    }
}

void OneLvPixMix::bwdUpdate(
    const float scAlpha,
    const float acAlpha,
    const float ccAlpha,
    const float thDist,
    const int maxRandSearchItr,
    const bool pixelWise
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
                //float cost = scAlpha * calcSptCost(target, ref, thDist) + acAlpha * calcAppCost(target, ref) + ccAlpha * calcConstrCost(target, ref) + calcDummyCost(ref);
                float cost = scAlpha * calcSptCost(target, ref, thDist) + acAlpha * calcAppCost(target, ref) + ccAlpha * calcConstrCost(target, ref) + calcDummyCost(ref);
                cost = isfinite(cost) ? cost : FLT_MAX;
                float costTop = FLT_MAX, costLeft = FLT_MAX;

                if (mMask[WO_BORDER](bottom) == 0 && mMask[WO_BORDER](bottomRef) != 0) {
                    costTop = scAlpha * calcSptCost(target, bottomRef, thDist) + acAlpha * calcAppCost(target, bottomRef) + ccAlpha * calcConstrCost(target, bottomRef) + calcDummyCost(bottomRef);
                    costTop = isfinite(costTop) ? costTop : FLT_MAX;
                }
                if (mMask[WO_BORDER](right) == 0 && mMask[WO_BORDER](rightRef) != 0) {
                    costLeft = scAlpha * calcSptCost(target, rightRef, thDist) + acAlpha * calcAppCost(target, rightRef) + ccAlpha * calcConstrCost(target, rightRef) + calcDummyCost(rightRef);
                    costLeft = isfinite(costLeft) ? costLeft : FLT_MAX;
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
                //do {
                //    refRand = getValidRandPos();
                //    costRand = scAlpha * calcSptCost(target, refRand, thDist) + acAlpha * calcAppCost(target, refRand) + ccAlpha * calcConstrCost(target, refRand) + calcDummyCost(refRand);
                //} while ((std::isinf(cost) && std::isinf(costRand)) || (!std::isinf(cost) && !std::isinf(costRand) && costRand >= cost/* && ++itrNum < maxRandSearchItrz*/));

                while (!isfinite(costRand)
                    || (costRand >= cost && itrNum < maxRandSearchItr)) {
                    refRand = getValidRandPos();

                    costRand = scAlpha * calcSptCost(target, refRand, thDist) + acAlpha * calcAppCost(target, refRand) + ccAlpha * calcConstrCost(target, refRand) + calcDummyCost(refRand);
                    costRand = isfinite(costRand) ? costRand : FLT_MAX;

                    ++itrNum;
                }

                if (!isfinite(cost) || costRand < cost) ptrPosMap[target[1]] = refRand;

                if (pixelWise) continue;

                cv::Mat_<cv::Vec3b> newPatch(windowSizeColorPatch, windowSizeColorPatch, cv::Vec3b::all(0));
                cv::Mat_<uchar> maskPatch(windowSizeColorPatch, windowSizeColorPatch, (uchar)0), discardPatch(windowSizeColorPatch, windowSizeColorPatch, (uchar)0);
                for (int i = -borderSizeColorPatch; i <= borderSizeColorPatch; i++) {
                    for (int j = -borderSizeColorPatch; j <= borderSizeColorPatch; j++) {
                        cv::Vec2i srcPos = ptrPosMap[target[1]] + cv::Vec2i(i, j);
                        cv::Vec2i targetPos = cv::Vec2i(target[0] + i, target[1] + j);

                        if (targetPos[0] < 0 || targetPos[1] < 0 || targetPos[0] >= mPosMap[WO_BORDER].rows || targetPos[1] >= mPosMap[WO_BORDER].cols) continue;
                        if (srcPos[0] < 0 || srcPos[1] < 0 || srcPos[0] >= mPosMap[WO_BORDER].rows || srcPos[1] >= mPosMap[WO_BORDER].cols) continue;

                        newPatch(borderSizeColorPatch + i, borderSizeColorPatch + j) = mColor[WO_BORDER](srcPos);
                        maskPatch(borderSizeColorPatch + i, borderSizeColorPatch + j) = mMask[WO_BORDER](srcPos);
                        discardPatch(borderSizeColorPatch + i, borderSizeColorPatch + j) = mDiscardings[WO_BORDER](srcPos);
                    }
                }

                cv::Mat_<uchar> newPatchMask = BottomRightBoundaryCut(newPatch, maskPatch, discardPatch, target);

                for (int i = -borderSizeColorPatch; i <= borderSizeColorPatch; i++) {
                    for (int j = -borderSizeColorPatch; j <= borderSizeColorPatch; j++) {
                        if (target[0] + i < 0 || target[1] + j < 0 || target[0] + i >= mPosMap[WO_BORDER].rows || target[1] + j >= mPosMap[WO_BORDER].cols) continue;
                        if (ptrPosMap[target[1]][0] + i < 0 || ptrPosMap[target[1]][1] + j < 0 || ptrPosMap[target[1]][0] + i >= mPosMap[WO_BORDER].rows || ptrPosMap[target[1]][1] + j >= mPosMap[WO_BORDER].cols) continue;

                        if (newPatchMask(borderSizeColorPatch + i, borderSizeColorPatch + j) != 0) {
                            mPosMap[WO_BORDER](target[0] + i, target[1] + j) = ptrPosMap[target[1]] + cv::Vec2i(i, j);  // texture quilting
                        }
                    }
                }
            }
        }
    }
}