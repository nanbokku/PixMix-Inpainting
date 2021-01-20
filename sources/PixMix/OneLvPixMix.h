#pragma once

#include <random>
#include <opencv2/opencv.hpp>

#include "Utilities.h"

class OneLvPixMix
{
public:
	OneLvPixMix();
	~OneLvPixMix();

	void init(const cv::Mat_<cv::Vec3b> &color, const cv::Mat_<uchar> &mask, const cv::Mat_<uchar>& discarding_area, const cv::Mat_<uchar>& gradient);
	// NOTE: Increasing "maxItr" and "maxRandSearchItr" will improve the quality of results but will decrease the speed as well
	void execute(const float scAlpha, const float acAlpha, const int maxItr, const int maxRandSearchItr, const float threshDist);

	cv::Mat_<cv::Vec3b> *getColorPtr();
	cv::Mat_<uchar> *getMaskPtr();
	cv::Mat_<uchar> *getDiscardingAreaPtr();
    cv::Mat_<uchar> *getGradientPtr();
	cv::Mat_<cv::Vec2i> *getPosMapPtr();

private:
	const int borderSize;
	const int borderSizePosMap;
	const int windowSize;

	enum { WO_BORDER = 0, W_BORDER = 1 };
	cv::Mat_<cv::Vec3b> mColor[2];
	cv::Mat_<uchar> mMask[2];
	cv::Mat_<uchar> mDiscardings[2];
    cv::Mat_<uchar> mGradient[2];
    cv::Mat_<uchar> mEdge[2];
	cv::Mat_<cv::Vec2i> mPosMap[2];	// current position map: f

	const cv::Vec2i toLeft;
	const cv::Vec2i toRight;
	const cv::Vec2i toUp;
	const cv::Vec2i toDown;
	std::vector<cv::Vec2i> vSptAdj;

	std::mt19937 mt;
	std::uniform_int_distribution<int> cRand;
	std::uniform_int_distribution<int> rRand;

	cv::Vec2i getValidRandPos();

	void inpaint();

	float calcSptCost(
		const cv::Vec2i &target,
		const cv::Vec2i &ref,
		float maxDist,		// tau_s
		float w = 0.125f	// 1.0f / 8.0f
	);
    float calcAppCost(
        const cv::Vec2i &target,
        const cv::Vec2i &ref,
        float w = 0.04f		// 1.0f / 25.0f
	);
	float calcConstrCost(
        const cv::Vec2i &target,
        const cv::Vec2i &ref
	);
    float calcDummyCost(
        const cv::Vec2i &ref
    );

    cv::Mat leftVerticalBoundaryCut(const cv::Mat_<cv::Vec3b>& newPatch, const cv::Vec2i& center);
    cv::Mat rightVerticalBoundaryCut(const cv::Mat_<cv::Vec3b>& newPatch, const cv::Vec2i& center);
    cv::Mat topHorizontalBoundaryCut(const cv::Mat_<cv::Vec3b>& newPatch, const cv::Vec2i& center);
    cv::Mat bottomHorizontalBoundaryCut(const cv::Mat_<cv::Vec3b>& newPatch, const cv::Vec2i& center);

	void fwdUpdate(
		const float scAlpha,
		const float acAlpha,
        const float ccAlpha,
		const float thDist,
		const int maxRandSearchItr
	);
	void bwdUpdate(
		const float scAlpha,
		const float acAlpha,
        const float ccAlpha,
		const float thDist,
		const int maxRandSearchItr
	);
};

inline cv::Mat_<cv::Vec3b> *OneLvPixMix::getColorPtr()
{
	return &(mColor[WO_BORDER]);
}
inline cv::Mat_<uchar> *OneLvPixMix::getMaskPtr()
{
	return &(mMask[WO_BORDER]);
}
inline cv::Mat_<uchar> *OneLvPixMix::getDiscardingAreaPtr()
{
	return &(mDiscardings[WO_BORDER]);
}
inline cv::Mat_<uchar> *OneLvPixMix::getGradientPtr()
{
    return &(mGradient[WO_BORDER]);
}
inline cv::Mat_<cv::Vec2i> *OneLvPixMix::getPosMapPtr()
{
	return &mPosMap[WO_BORDER];
}

inline cv::Vec2i OneLvPixMix::getValidRandPos()
{
	cv::Vec2i p;
	do {
		p = cv::Vec2i(rRand(mt), cRand(mt));
	} while (mMask[WO_BORDER](p) != 255 || mDiscardings[WO_BORDER](p) != 255);

	return p;
}